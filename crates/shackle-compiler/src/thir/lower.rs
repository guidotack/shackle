//! Functionality for converting HIR nodes into THIR nodes.
//!
//! The following is performed during lowering:
//! - Assignment items are moved into declarations/constraints
//! - Destructuring declarations are rewritten as separate declarations
//! - Destructuring in generators is rewritten into a where clause
//! - Type alias items removed as they have been resolved
//! - 2D array literals are re-written using `array2d` calls
//! - Indexed array literals are re-written using `arrayNd` calls
//! - Array slicing is re-written using calls to `slice_Xd`
//! - Tuple/record access into arrays of structs are rewritten using a
//!   comprehension accessing the inner value

use std::{collections::hash_map::Entry, fmt::format, sync::Arc};

use rustc_hash::FxHashMap;

use super::{
	db::{Intermediate, Thir},
	source::Origin,
	*,
};
use crate::{
	constants::IdentifierRegistry,
	hir::{
		self,
		ids::{EntityRef, ExpressionRef, ItemRef, LocalItemRef, NodeRef, PatternRef},
		PatternTy, TypeResult,
	},
	ty::{EnumRef, Ty, TyData},
	utils::{arena::ArenaIndex, impl_enum_from, maybe_grow_stack},
};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum DeclOrConstraint {
	Declaration(DeclarationId),
	Constraint(ConstraintId),
}

impl_enum_from!(DeclOrConstraint::Declaration(DeclarationId));
impl_enum_from!(DeclOrConstraint::Constraint(ConstraintId));

impl From<DeclOrConstraint> for LetItem {
	fn from(d: DeclOrConstraint) -> Self {
		match d {
			DeclOrConstraint::Constraint(c) => LetItem::Constraint(c),
			DeclOrConstraint::Declaration(d) => LetItem::Declaration(d),
		}
	}
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum LoweredAnnotation {
	Items(Vec<DeclOrConstraint>),
	Expression(Expression),
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum LoweredIdentifier {
	ResolvedIdentifier(ResolvedIdentifier),
	Callable(Callable),
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ClassObjectsInfo {
	class_objects: DeclarationId,
	class_enum_member: Option<u32>,
}

/// Collects HIR items and lowers them to THIR
struct ItemCollector<'a> {
	db: &'a dyn Thir,
	ids: &'a IdentifierRegistry,
	resolutions: FxHashMap<PatternRef, LoweredIdentifier>,
	model: Model,
	type_alias_expressions: FxHashMap<ExpressionRef, DeclarationId>,
	deferred: Vec<(FunctionId, ItemRef)>,
	class_objects: FxHashMap<PatternRef, Vec<ClassObjectsInfo>>,
}

impl<'a> ItemCollector<'a> {
	/// Create a new item collector
	pub fn new(
		db: &'a dyn Thir,
		ids: &'a IdentifierRegistry,
		entity_counts: &hir::db::EntityCounts,
	) -> Self {
		Self {
			db,
			ids,
			resolutions: FxHashMap::default(),
			model: Model::with_capacities(&entity_counts.clone().into()),
			type_alias_expressions: FxHashMap::default(),
			deferred: Vec::new(),
			class_objects: FxHashMap::default(),
		}
	}

	/// Collect an item
	pub fn collect_item(&mut self, item: ItemRef) {
		let model = item.model(self.db.upcast());
		let local_item = item.local_item_ref(self.db.upcast());
		match local_item {
			LocalItemRef::Annotation(a) => {
				self.collect_annotation(item, &model[a]);
			}
			LocalItemRef::Assignment(a) => self.collect_assignment(item, &model[a]),
			LocalItemRef::Constraint(c) => {
				self.collect_constraint(item, &model[c], &model[c].data, true);
			}
			LocalItemRef::Declaration(d) => {
				self.collect_declaration(item, &model[d], &model[d].data, true);
			}
			LocalItemRef::Enumeration(e) => {
				self.collect_enumeration(item, &model[e]);
			}
			LocalItemRef::EnumAssignment(a) => self.collect_enumeration_assignment(item, &model[a]),
			LocalItemRef::Function(f) => {
				self.collect_function(item, &model[f]);
			}
			LocalItemRef::Output(o) => {
				self.collect_output(item, &model[o]);
			}
			LocalItemRef::Solve(s) => self.collect_solve(item, &model[s]),
			LocalItemRef::TypeAlias(t) => self.collect_type_alias(item, &model[t]),
			LocalItemRef::Class(c) => self.collect_class(item, &model[c]),
		}
	}

	/// Collect a class declaration item
	pub fn collect_class(&mut self, item: ItemRef, c: &hir::Item<hir::Class>) {
		let types = self.db.lookup_item_types(item);
		let analysis = self.db.class_analysis();
		let class_occurrences = analysis
			.map_class_to_constrs
			.get(&PatternRef::new(item, c.pattern));
		let class_subclasses = analysis
			.map_class_to_subclasses
			.get(&PatternRef::new(item, c.pattern));
		let n_occurrences = class_occurrences.map_or(0, |x| x.len());
		let n_subclasses = class_subclasses.map_or(0, |x| x.len());
		let superclasses = match types[c.pattern] {
			PatternTy::ClassDecl {
				defining_set_ty,
				input_record_ty,
			} => defining_set_ty
				.class_type(self.db.upcast())
				.unwrap()
				.superclasses(self.db.upcast())
				.collect::<Vec<_>>(),
			_ => unreachable!(),
		};
		let mut constructors = Vec::with_capacity(n_occurrences + n_subclasses);
		let mut class_objects =
			std::iter::repeat_with(|| Vec::with_capacity(n_occurrences + n_subclasses))
				.take(superclasses.len())
				.collect::<Vec<Vec<DeclarationId>>>();
		let mut need_potential = false;
		let mut objects_are_var = false;
		for (patternref, ci_idx) in class_occurrences.iter().flat_map(|x| x.iter()) {
			let m = patternref.item().model(self.db.upcast());
			let decl_types = self.db.lookup_item_types(patternref.item());
			match patternref.item().local_item_ref(self.db.upcast()) {
				LocalItemRef::Declaration(arena_index) => {
					match &decl_types[patternref.pattern()] {
						PatternTy::Variable(ty) => {
							if ty.inst(self.db.upcast()) == Some(VarType::Var) {
								objects_are_var = true;
							}
							if ty.opt(self.db.upcast()) == Some(OptType::Opt) {
								need_potential = true;
							}
							let data = &m[arena_index].data;
							let dt = &data[m[arena_index].declared_type];
							match dt {
								hir::Type::Set {
									inst: VarType::Var,
									opt,
									cardinality: Some(card),
									element,
								} => {
									need_potential = true;
									let mut collector = ExpressionCollector::new(
										self,
										data,
										patternref.item(),
										&decl_types,
									);
									let exp = collector.collect_expression(*card);
									let origin = exp.origin();
									let max_call = LookupCall {
										function: collector.parent.ids.max.into(),
										arguments: vec![exp],
									};
									let max_exp = Expression::new(
										collector.parent.db,
										&collector.parent.model,
										origin,
										max_call,
									);
									let one_exp = Expression::new(
										collector.parent.db,
										&collector.parent.model,
										origin,
										IntegerLiteral(1),
									);
									let dotdot_call = LookupCall {
										function: collector.parent.ids.dot_dot.into(),
										arguments: vec![one_exp, max_exp],
									};
									let dotdot_exp = Expression::new(
										collector.parent.db,
										&collector.parent.model,
										origin,
										dotdot_call,
									);

									let enum_constr_domain = Domain::bounded(
										collector.parent.db,
										origin,
										VarType::Par,
										OptType::NonOpt,
										dotdot_exp,
									);

									let decl = Declaration::new(false, enum_constr_domain.clone());
									let idx = collector
										.parent
										.model
										.add_declaration(Item::new(decl, origin));
									constructors.push(Constructor {
										name: Some(Identifier::new(
											format!(
												"{}_Class",
												m[arena_index].data[m[arena_index].pattern]
													.identifier()
													.unwrap()
													.lookup(self.db.upcast())
											),
											self.db.upcast(),
										)),
										parameters: Some(vec![idx]),
									});

									for (super_c_idx, super_c) in superclasses.iter().enumerate() {
										let element = Domain::record(
											self.db,
											origin,
											OptType::NonOpt,
											super_c.attributes.iter().map(|(ident, ty)| {
												(*ident, {
													let class_item =
														super_c.pattern(self.db.upcast()).item();
													let class_decl_types =
														self.db.lookup_item_types(class_item);
													let m = class_item.model(self.db.upcast());
													match class_item
														.local_item_ref(self.db.upcast())
													{
														LocalItemRef::Class(lir_c) => m[lir_c]
															.items
															.iter()
															.find_map(|item| match item {
																hir::ClassItem::Declaration(d) => {
																	if m[lir_c].data[d.pattern]
																		.identifier() == Some(*ident)
																	{
																		let mut collector = ExpressionCollector::new(
																							self,
																							&m[lir_c].data,
																							class_item,
																							&class_decl_types,
																						);
																		Some(
																			collector
																				.collect_domain(
																					d.declared_type,
																					ty.make_var(
																						collector
																							.parent
																							.db
																							.upcast(
																							),
																					)
																					.unwrap(),
																					false,
																				),
																		)
																	} else {
																		None
																	}
																}
																_ => None,
															}),
														_ => unreachable!(),
													}
													.unwrap()
												})
											}),
										);

										let dom = Domain::array(
											self.db,
											origin,
											OptType::NonOpt,
											enum_constr_domain.clone(),
											element,
										);

										let mut obj_decl = Declaration::new(true, dom);
										let field_name = patternref
											.identifier(self.db.upcast())
											.unwrap()
											.lookup(self.db.upcast());

										obj_decl.set_name(Identifier::new(
											format!(
												"{}_{}_{}_objects",
												super_c.pretty_print(self.db.upcast()),
												PatternRef::new(item, c.pattern)
													.identifier(self.db.upcast())
													.unwrap()
													.lookup(self.db.upcast()),
												field_name
											),
											self.db.upcast(),
										));
										let obj_idx =
											self.model.add_declaration(Item::new(obj_decl, origin));
										class_objects[super_c_idx].push(obj_idx);
									}
								}
								hir::Type::Set {
									inst: VarType::Var,
									opt,
									cardinality: None,
									element,
								} => unreachable!("Var set without cardinality not supportd (should be a type error)"),
								hir::Type::New { inst: VarType::Par, opt: OptType::Opt, .. } |
								hir::Type::Set {
									inst: VarType::Par,
									..
								} => {
									let isSet = matches!(dt, hir::Type::Set { .. });

									// Compute type of the input array.
									// The type is a record that has all par fields without right hand side of this class and all superclasses.
									let record_ty = match types.get_pattern(c.pattern).unwrap() {
										PatternTy::ClassDecl {
											input_record_ty, ..
										} => Ty::array(
											self.db.upcast(),
											Ty::par_int(self.db.upcast()),
											*input_record_ty,
										)
										.unwrap(),
										_ => unreachable!(),
									};
									let dom = Domain::unbounded(self.db, item, record_ty);
									let mut decl = Declaration::new(true, dom);
									let field_name = patternref
										.identifier(self.db.upcast())
										.unwrap()
										.lookup(self.db.upcast());
									if isSet {
										decl.set_name(Identifier::new(
											format!("{}_inputs", field_name),
											self.db.upcast(),
										));
										if let Some(def) = m[arena_index].definition {
											let mut collector = ExpressionCollector::new(
												self,
												data,
												patternref.item(),
												&decl_types,
											);
											decl.set_definition(collector.collect_expression(def));
										}
									} else {
										decl.set_name(Identifier::new(
											format!("{}_inputs_set", field_name),
											self.db.upcast(),
										));

										let opt_record_ty = match types.get_pattern(c.pattern).unwrap() {
											PatternTy::ClassDecl {
												input_record_ty, ..
											} => 
												input_record_ty.with_opt(self.db.upcast(), OptType::Opt),
											_ => unreachable!(),
										};
										let opt_dom = Domain::unbounded(self.db, item, opt_record_ty);
										let mut opt_decl = Declaration::new(true, opt_dom);
										opt_decl.set_name(Identifier::new(
											format!("{}_inputs", field_name),
											self.db.upcast(),
										));

										if let Some(def) = m[arena_index].definition {
											let mut collector = ExpressionCollector::new(
												self,
												data,
												patternref.item(),
												&decl_types,
											);
											opt_decl.set_definition(collector.collect_expression(def));
										}
										let opt_decl_idx =
										self.model.add_declaration(Item::new(opt_decl, item));
										
										let occurs_arg = Expression::new(
											self.db,
											&self.model,
											item,
											ResolvedIdentifier::Declaration(opt_decl_idx),
										);
										let occurs_call = Expression::new(
											self.db,
											&self.model,
											item,
											LookupCall {
												function: self.ids.occurs.into(),
												arguments: vec![occurs_arg.clone()],
											},
										);
										let deopt_call = Expression::new(
											self.db,
											&self.model,
											item,
											LookupCall {
												function: self.ids.deopt.into(),
												arguments: vec![occurs_arg],
											},
										);
										let array_deopt = Expression::new(
											self.db,
											&self.model,
											item,
											ArrayLiteral(vec![deopt_call]),
										);
										let array_empty = Expression::new(
											self.db,
											&self.model,
											item,
											ArrayLiteral(vec![]),
										);

										decl.set_definition(Expression::new(
											self.db,
											&self.model,
											item,
											IfThenElse{ branches: vec![Branch { condition: occurs_call, result: array_deopt }], else_result: Box::new(array_empty) },
										))
	
									}
									let decl_idx =
										self.model.add_declaration(Item::new(decl, item));
									let inputs_expr = Expression::new(
										self.db,
										&self.model,
										item,
										ResolvedIdentifier::Declaration(decl_idx),
									);
									let len_exp = Expression::new(
										self.db,
										&self.model,
										item,
										LookupCall {
											function: self.ids.length.into(),
											arguments: vec![inputs_expr.clone()],
										},
									);

									let one_exp = Expression::new(
										self.db,
										&self.model,
										item,
										IntegerLiteral(1),
									);
									let dotdot_call = LookupCall {
										function: self.ids.dot_dot.into(),
										arguments: vec![one_exp, len_exp],
									};
									let dotdot_exp =
										Expression::new(self.db, &self.model, item, dotdot_call);
									let decl: Declaration<_> = Declaration::new(
										false,
										Domain::bounded(
											self.db,
											item,
											VarType::Par,
											OptType::NonOpt,
											dotdot_exp,
										),
									);
									let idx = self.model.add_declaration(Item::new(decl, item));
									constructors.push(Constructor {
										name: Some(Identifier::new(
											format!(
												"{}_{}",
												PatternRef::new(item, c.pattern)
													.identifier(self.db.upcast())
													.unwrap()
													.lookup(self.db.upcast()),
												field_name
											),
											self.db.upcast(),
										)),
										parameters: Some(vec![idx]),
									});

									let origin = inputs_expr.origin();
									let inputs_elem_ty =
										inputs_expr.ty().elem_ty(self.db.upcast()).unwrap();
									for (super_c_idx, super_c) in superclasses.iter().enumerate() {
										let gen_decl = Declaration::new(
											false,
											Domain::unbounded(self.db, origin, inputs_elem_ty),
										);
										let gen_idx =
											self.model.add_declaration(Item::new(gen_decl, origin));

										let comp_exp = RecordLiteral(
											super_c
												.attributes
												.iter()
												.map(|(ident, ty)| {
													(
														*ident,
														if ty.inst(self.db.upcast())
															== Some(VarType::Par)
														{
															Expression::new(
																self.db,
																&self.model,
																origin,
																RecordAccess {
																	record: Box::new(
																		Expression::new(
																			self.db,
																			&self.model,
																			origin,
																			gen_idx,
																		),
																	),
																	field: *ident,
																},
															)
														} else {
															let class_item = super_c
																.pattern(self.db.upcast())
																.item();
															let class_decl_types = self
																.db
																.lookup_item_types(class_item);
															let m =
																class_item.model(self.db.upcast());
															let decl_idx = match class_item
																.local_item_ref(self.db.upcast())
															{
																LocalItemRef::Class(lir_c) => {
																	m[lir_c].items.iter().find_map(
																		|item| {
																			match item {
																				hir::ClassItem::Declaration(d) => {
																					if m[lir_c].data[d.pattern].identifier() == Some(*ident) {
																						let mut collector = ExpressionCollector::new(
																							self,
																							&m[lir_c].data,
																							class_item,
																							&class_decl_types,
																						);
																						let domain = collector.collect_domain(d.declared_type, *ty, false);
																						let decl = Declaration::new(false, domain);
																						let idx = self.model.add_declaration(Item::new(decl, class_item));
																						Some(idx)
																					} else {
																						None
																					}
																				}
																				_ => None,
																			}
																		},
																	)
																}
																_ => unreachable!(),
															}
															.unwrap();
															Expression::new(
																		self.db,
																		&self.model,
																		origin,
																		Let {
																			items: vec![LetItem::Declaration(decl_idx)],
																			in_expression: Box::new(Expression::new(
																				self.db,
																				&self.model,
																				origin,
																				ResolvedIdentifier::Declaration(decl_idx),
																			)),
																		}
																	)
														},
													)
												})
												.collect(),
										);
										let comp = Expression::new(
											self.db,
											&self.model,
											origin,
											ArrayComprehension {
												indices: None,
												generators: vec![Generator::Iterator {
													declarations: vec![gen_idx],
													collection: inputs_expr.clone(),
													where_clause: None,
												}],
												template: Box::new(Expression::new(
													self.db,
													&self.model,
													origin,
													comp_exp,
												)),
											},
										);
										let mut comp_decl =
											Declaration::from_expression(self.db, true, comp);
										comp_decl.set_name(Identifier::new(
											format!(
												"{}_{}_{}_objects",
												super_c.pretty_print(self.db.upcast()),
												PatternRef::new(item, c.pattern)
													.identifier(self.db.upcast())
													.unwrap()
													.lookup(self.db.upcast()),
												field_name
											),
											self.db.upcast(),
										));
										let comp_idx = self
											.model
											.add_declaration(Item::new(comp_decl, origin));
										class_objects[super_c_idx].push(comp_idx);
									}
								}
								hir::Type::New { inst: VarType::Par, opt: OptType::NonOpt, domain } => {
									// Compute type of the input declaration (same as for sets, but not an array)
									let record_ty = match types.get_pattern(c.pattern).unwrap() {
										PatternTy::ClassDecl {
											input_record_ty, ..
										} => *input_record_ty,
										_ => unreachable!(),
									};
									let dom = Domain::unbounded(self.db, item, record_ty);
									let mut decl = Declaration::new(true, dom);
									let field_name = patternref
										.identifier(self.db.upcast())
										.unwrap()
										.lookup(self.db.upcast());
									decl.set_name(Identifier::new(
										format!("{}_input", field_name),
										self.db.upcast(),
									));
									if let Some(def) = m[arena_index].definition {
										let mut collector = ExpressionCollector::new(
											self,
											data,
											patternref.item(),
											&decl_types,
										);
										decl.set_definition(collector.collect_expression(def));
									}
									let decl_idx =
										self.model.add_declaration(Item::new(decl, item));

									constructors.push(Constructor {
										name: Some(Identifier::new(
											format!(
												"{}_Class",
												m[arena_index].data[m[arena_index].pattern]
													.identifier()
													.unwrap()
													.lookup(self.db.upcast())
											),
											self.db.upcast(),
										)),
										parameters: None,
									});
									let origin = item;
									for (super_c_idx, super_c) in superclasses.iter().enumerate() {
										let comp_exp = RecordLiteral(
											super_c
												.attributes
												.iter()
												.map(|(ident, ty)| {
													(
														*ident,
														if ty.inst(self.db.upcast())
															== Some(VarType::Par)
														{
															Expression::new(
																self.db,
																&self.model,
																origin,
																RecordAccess {
																	record: Box::new(
																		Expression::new(
																			self.db,
																			&self.model,
																			origin,
																			decl_idx,
																		),
																	),
																	field: *ident,
																},
															)
														} else {
															let class_item = super_c
																.pattern(self.db.upcast())
																.item();
															let class_decl_types = self
																.db
																.lookup_item_types(class_item);
															let m =
																class_item.model(self.db.upcast());
															let decl_idx = match class_item
																.local_item_ref(self.db.upcast())
															{
																LocalItemRef::Class(lir_c) => {
																	m[lir_c].items.iter().find_map(
																		|item| {
																			match item {
																				hir::ClassItem::Declaration(d) => {
																					if m[lir_c].data[d.pattern].identifier() == Some(*ident) {
																						let mut collector = ExpressionCollector::new(
																							self,
																							&m[lir_c].data,
																							class_item,
																							&class_decl_types,
																						);
																						let domain = collector.collect_domain(d.declared_type, *ty, false);
																						let decl = Declaration::new(false, domain);
																						let idx = self.model.add_declaration(Item::new(decl, class_item));
																						Some(idx)
																					} else {
																						None
																					}
																				}
																				_ => None,
																			}
																		},
																	)
																}
																_ => unreachable!(),
															}
															.unwrap();
															Expression::new(
																		self.db,
																		&self.model,
																		origin,
																		Let {
																			items: vec![LetItem::Declaration(decl_idx)],
																			in_expression: Box::new(Expression::new(
																				self.db,
																				&self.model,
																				origin,
																				ResolvedIdentifier::Declaration(decl_idx),
																			)),
																		}
																	)
														},
													)
												})
												.collect(),
										);

										let comp_exp_exp = 
										Expression::new(
											self.db,
											&self.model,
											origin,
											ArrayLiteral(
												vec![Expression::new(
													self.db,
													&self.model,
													origin,
													comp_exp,
													)
												]
											),
										);

										let mut comp_decl =
											Declaration::from_expression(self.db, true, comp_exp_exp);
										comp_decl.set_name(Identifier::new(
											format!(
												"{}_{}_{}_object",
												super_c.pretty_print(self.db.upcast()),
												PatternRef::new(item, c.pattern)
													.identifier(self.db.upcast())
													.unwrap()
													.lookup(self.db.upcast()),
												field_name
											),
											self.db.upcast(),
										));
										let comp_idx = self
											.model
											.add_declaration(Item::new(comp_decl, origin));
										class_objects[super_c_idx].push(comp_idx);
									}
								}
								hir::Type::New { inst: VarType::Var, opt, domain } => {
									let origin = patternref.item();

									for (super_c_idx, super_c) in superclasses.iter().enumerate() {
										let element = Domain::record(
											self.db,
											origin,
											OptType::NonOpt,
											super_c.attributes.iter().map(|(ident, ty)| {
												(*ident, {
													let class_item =
														super_c.pattern(self.db.upcast()).item();
													let class_decl_types =
														self.db.lookup_item_types(class_item);
													let m = class_item.model(self.db.upcast());
													match class_item
														.local_item_ref(self.db.upcast())
													{
														LocalItemRef::Class(lir_c) => m[lir_c]
															.items
															.iter()
															.find_map(|item| match item {
																hir::ClassItem::Declaration(d) => {
																	if m[lir_c].data[d.pattern]
																		.identifier() == Some(*ident)
																	{
																		let mut collector = ExpressionCollector::new(
																							self,
																							&m[lir_c].data,
																							class_item,
																							&class_decl_types,
																						);
																		Some(
																			collector
																				.collect_domain(
																					d.declared_type,
																					ty.make_var(
																						collector
																							.parent
																							.db
																							.upcast(
																							),
																					)
																					.unwrap(),
																					false,
																				),
																		)
																	} else {
																		None
																	}
																}
																_ => None,
															}),
														_ => unreachable!(),
													}
													.unwrap()
												})
											}),
										);

										let one_one = Expression::new(
											self.db,
											&self.model,
											origin,
											SetLiteral(vec![Expression::new(
												self.db,
												&self.model,
												origin,
												IntegerLiteral(1),
											)]),
										);

										let dom = Domain::array(
											self.db,
											origin,
											OptType::NonOpt,
											Domain::bounded(self.db, origin, VarType::Par, OptType::NonOpt, one_one),
											element,
										);

										let mut obj_decl = Declaration::new(true, dom);
										let field_name = patternref
											.identifier(self.db.upcast())
											.unwrap()
											.lookup(self.db.upcast());

										obj_decl.set_name(Identifier::new(
											format!(
												"{}_{}_{}_objects",
												super_c.pretty_print(self.db.upcast()),
												PatternRef::new(item, c.pattern)
													.identifier(self.db.upcast())
													.unwrap()
													.lookup(self.db.upcast()),
												field_name
											),
											self.db.upcast(),
										));
										let obj_idx =
											self.model.add_declaration(Item::new(obj_decl, origin));
										class_objects[super_c_idx].push(obj_idx);
									}

									constructors.push(Constructor {
										name: Some(Identifier::new(
											format!(
												"{}_singleton",
												m[arena_index].data[m[arena_index].pattern]
													.identifier()
													.unwrap()
													.lookup(self.db.upcast())
											),
											self.db.upcast(),
										)),
										parameters: None,
									});
								}
								_ => unreachable!()
							}
						}
						_ => unreachable!(),
					}
				}
				LocalItemRef::Class(arena_index) => {
					let ci_item = &m[arena_index].items[*ci_idx];
				}
				_ => unreachable!(),
			}
		}

		let class_ident = c.data[c.pattern]
			.identifier()
			.unwrap()
			.lookup(self.db.upcast());
		for patternref in class_subclasses.iter().flat_map(|x| x.iter()) {
			let ident = patternref.identifier(self.db.upcast()).unwrap();

			let ident_expr = match self.resolutions.entry(*patternref) {
				Entry::Occupied(e) => match e.get() {
					LoweredIdentifier::ResolvedIdentifier(ResolvedIdentifier::Enumeration(i)) => *i,
					_ => unreachable!(),
				},
				Entry::Vacant(e) => {
					let name = Identifier::new(
						format!("{}_Potential", ident.lookup(self.db.upcast())),
						self.db.upcast(),
					);
					let enumeration = Enumeration::new(EnumRef::introduce(self.db.upcast(), name));
					let idx = self.model.add_enumeration(Item::new(enumeration, item));
					let resolved_ident = LoweredIdentifier::ResolvedIdentifier(idx.into());
					e.insert(resolved_ident);
					idx
				}
			};

			let decl = Declaration::new(
				false,
				Domain::bounded(
					self.db,
					item,
					VarType::Par,
					OptType::NonOpt,
					Expression::new(self.db, &self.model, item, ident_expr),
				),
			);
			let idx = self.model.add_declaration(Item::new(decl, item));

			constructors.push(Constructor {
				name: Some(Identifier::new(
					format!("{}_in_{}", ident.lookup(self.db.upcast()), class_ident),
					self.db.upcast(),
				)),
				parameters: Some(vec![idx]),
			});

			match self.class_objects.entry(*patternref) {
				Entry::Occupied(mut e) => {
					for (subclass_o_idx, subclass_o) in e.get_mut().iter_mut().skip(1).enumerate() {
						subclass_o.class_enum_member =
							Some(class_objects[subclass_o_idx].len() as u32);
						class_objects[subclass_o_idx].push(subclass_o.class_objects);
					}
				}
				Entry::Vacant(e) => {
					let mut subclass_objects = Vec::with_capacity(superclasses.len() + 1);

					let mut add_class_objects_decl =
						|subclass_item: ItemRef, subclass_name: Identifier| {
							let m = subclass_item.model(self.db.upcast());

							let subclass_record_ty =
								match subclass_item.local_item_ref(self.db.upcast()) {
									LocalItemRef::Class(sc) => {
										let types = self.db.lookup_item_types(subclass_item);
										let fields = match types[m[sc].pattern] {
											PatternTy::ClassDecl {
												defining_set_ty,
												input_record_ty,
											} => {
												defining_set_ty
													.class_type(self.db.upcast())
													.unwrap()
													.attributes
											}
											_ => unreachable!(),
										};
										Ty::array(
											self.db.upcast(),
											Ty::par_int(self.db.upcast()),
											Ty::record(self.db.upcast(), fields),
										)
										.unwrap()
									}
									_ => unreachable!(),
								};
							let mut subclass_decl = Declaration::new(
								true,
								Domain::unbounded(self.db, subclass_item, subclass_record_ty),
							);
							subclass_decl.set_name(subclass_name);

							self.model
								.add_declaration(Item::new(subclass_decl, subclass_item))
						};

					let subclass_name = Identifier::new(
						format!("{}_Objects", ident.lookup(self.db.upcast())),
						self.db.upcast(),
					);

					let subclass_objects_idx =
						add_class_objects_decl(patternref.item(), subclass_name);
					subclass_objects.push(ClassObjectsInfo {
						class_objects: subclass_objects_idx,
						class_enum_member: None,
					});

					for (super_c_idx, super_c) in superclasses.iter().enumerate() {
						let superclass_name = Identifier::new(
							format!(
								"{}_{}_Objects",
								super_c.pretty_print(self.db.upcast()),
								ident.lookup(self.db.upcast())
							),
							self.db.upcast(),
						);
						let superclass_objects_idx = add_class_objects_decl(
							super_c.pattern(self.db.upcast()).item(),
							superclass_name,
						);

						subclass_objects.push(ClassObjectsInfo {
							class_objects: subclass_objects_idx,
							class_enum_member: Some(class_objects[super_c_idx].len() as u32),
						});
						class_objects[super_c_idx].push(superclass_objects_idx);
					}

					e.insert(subclass_objects);
				}
			};
		}

		let class_pattern_ref = PatternRef::new(item, c.pattern);
		let class_enum_id = match self.resolutions.entry(class_pattern_ref) {
			Entry::Occupied(e) => match e.get() {
				LoweredIdentifier::ResolvedIdentifier(ResolvedIdentifier::Enumeration(i)) => {
					self.model[*i].set_definition(constructors);
					*i
				}
				_ => unreachable!(),
			},
			Entry::Vacant(e) => {
				let name = Identifier::new(format!("{}_Potential", class_ident), self.db.upcast());
				let mut enumeration = Enumeration::new(EnumRef::introduce(self.db.upcast(), name));
				enumeration.set_definition(constructors);
				let idx = self.model.add_enumeration(Item::new(enumeration, item));
				e.insert(LoweredIdentifier::ResolvedIdentifier(idx.into()));
				idx
			}
		};

		let class_object_decls = self
			.class_objects
			.entry(class_pattern_ref)
			.or_insert_with(|| {
				let mut subclass_objects = Vec::with_capacity(superclasses.len());

				let mut add_class_objects_decl =
					|subclass_item: ItemRef, subclass_name: Identifier| {
						let m = subclass_item.model(self.db.upcast());

						let subclass_record_ty =
							match subclass_item.local_item_ref(self.db.upcast()) {
								LocalItemRef::Class(sc) => {
									let types = self.db.lookup_item_types(subclass_item);
									let fields = match types[m[sc].pattern] {
										PatternTy::ClassDecl {
											defining_set_ty,
											input_record_ty,
										} => {
											defining_set_ty
												.class_type(self.db.upcast())
												.unwrap()
												.attributes
										}
										_ => unreachable!(),
									};
									Ty::array(
										self.db.upcast(),
										Ty::par_int(self.db.upcast()),
										Ty::record(self.db.upcast(), fields),
									)
									.unwrap()
								}
								_ => unreachable!(),
							};
						let mut subclass_decl = Declaration::new(
							true,
							Domain::unbounded(self.db, subclass_item, subclass_record_ty),
						);
						subclass_decl.set_name(subclass_name);

						self.model
							.add_declaration(Item::new(subclass_decl, subclass_item))
					};

				let subclass_name =
					Identifier::new(format!("{}_Objects", class_ident), self.db.upcast());

				let subclass_objects_idx = add_class_objects_decl(item, subclass_name);
				subclass_objects.push(ClassObjectsInfo {
					class_objects: subclass_objects_idx,
					class_enum_member: None,
				});

				for super_c in superclasses.iter().skip(1) {
					let superclass_name = Identifier::new(
						format!(
							"{}_{}_Objects",
							super_c.pretty_print(self.db.upcast()),
							class_ident
						),
						self.db.upcast(),
					);
					let superclass_objects_idx = add_class_objects_decl(
						super_c.pattern(self.db.upcast()).item(),
						superclass_name,
					);

					subclass_objects.push(ClassObjectsInfo {
						class_objects: superclass_objects_idx,
						class_enum_member: None,
					});
				}

				subclass_objects
			});

		for (class_obj_decl, class_obj_occurrence_decls) in
			class_object_decls.iter().zip(class_objects)
		{
			if class_obj_occurrence_decls.is_empty() {
				let empty_array =
					Expression::new(self.db, &self.model, item, ArrayLiteral(Vec::new()));
				self.model[class_obj_decl.class_objects].set_definition(empty_array);
			} else {
				let class_obj_rhs = class_obj_occurrence_decls.iter().skip(1).fold(
					Expression::new(self.db, &self.model, item, class_obj_occurrence_decls[0]),
					|acc, element| {
						Expression::new(
							self.db,
							&self.model,
							item,
							LookupCall {
								function: self.ids.plus_plus.into(),
								arguments: vec![
									acc,
									Expression::new(self.db, &self.model, item, *element),
								],
							},
						)
					},
				);
				self.model[class_obj_decl.class_objects].set_definition(class_obj_rhs);
			}
		}

		// let mut record_fields = Vec::new();
		// for item in c.items.iter() {
		// 	match item {
		// 		hir::ClassItem::Declaration(d) => {

		// 		}
		// 		_ => {}
		// 	}
		// }
		// let record_domain = Domain::record(self.db, item, OptType::NonOpt, record_fields);
	}

	/// Collect an annotation item
	pub fn collect_annotation(
		&mut self,
		item: ItemRef,
		a: &hir::Item<hir::Annotation>,
	) -> AnnotationId {
		let types = self.db.lookup_item_types(item);
		let ty = &types[a.constructor_pattern()];
		match (&a.constructor, ty) {
			(hir::Constructor::Atom { pattern }, PatternTy::AnnotationAtom) => {
				let annotation = Annotation::new(
					a.data[*pattern]
						.identifier()
						.expect("Annotation must have identifier pattern"),
				);
				let idx = self.model.add_annotation(Item::new(annotation, item));
				self.resolutions.insert(
					PatternRef::new(item, *pattern),
					LoweredIdentifier::ResolvedIdentifier(idx.into()),
				);
				idx
			}
			(
				hir::Constructor::Function {
					constructor,
					destructor,
					parameters: params,
				},
				PatternTy::AnnotationConstructor(fn_entry),
			) => {
				let mut parameters = Vec::with_capacity(fn_entry.overload.params().len());
				for (param, ty) in params.iter().zip(fn_entry.overload.params()) {
					let mut collector = ExpressionCollector::new(self, &a.data, item, &types);
					let domain = collector.collect_domain(param.declared_type, *ty, false);
					let mut param_decl = Declaration::new(false, domain);
					// Ignore destructuring and recording resolution for now since these can't have bodies which refer
					// to parameters anyway
					if let Some(p) = param.pattern {
						if let Some(i) = a.data[p].identifier() {
							param_decl.set_name(i);
						}
					}
					let idx = self.model.add_declaration(Item::new(param_decl, item));
					parameters.push(idx);
				}
				let mut annotation = Annotation::new(
					a.data[*constructor]
						.identifier()
						.expect("Annotation must have identifier pattern"),
				);
				annotation.parameters = Some(parameters);
				let idx = self.model.add_annotation(Item::new(annotation, item));
				self.resolutions.insert(
					PatternRef::new(item, *constructor),
					LoweredIdentifier::Callable(Callable::Annotation(idx)),
				);
				self.resolutions.insert(
					PatternRef::new(item, *destructor),
					LoweredIdentifier::Callable(Callable::AnnotationDestructure(idx)),
				);
				idx
			}
			_ => unreachable!(),
		}
	}

	/// Collect an assignment item
	pub fn collect_assignment(&mut self, item: ItemRef, a: &hir::Item<hir::Assignment>) {
		let db = self.db;
		let types = db.lookup_item_types(item);
		let res = types.name_resolution(a.assignee).unwrap();
		let decl = match &self.resolutions[&res] {
			LoweredIdentifier::ResolvedIdentifier(ResolvedIdentifier::Declaration(d)) => *d,
			_ => unreachable!(),
		};
		if self.model[decl].definition().is_some() {
			// Turn subsequent assignment items into equality constraints
			let mut collector = ExpressionCollector::new(self, &a.data, item, &types);
			let call = LookupCall {
				function: collector.parent.ids.eq.into(),
				arguments: vec![
					collector.collect_expression(a.assignee),
					collector.collect_expression(a.definition),
				],
			};
			let constraint = Constraint::new(
				true,
				Expression::new(db, &collector.parent.model, item, call),
			);
			self.model.add_constraint(Item::new(constraint, item));
		} else {
			let mut declaration = self.model[decl].clone();
			let mut collector = ExpressionCollector::new(self, &a.data, item, &types);
			let def = collector.collect_expression(a.definition);
			declaration.set_definition(def);
			self.model[decl] = declaration;
		}
	}

	/// Collect a constraint item
	pub fn collect_constraint(
		&mut self,
		item: ItemRef,
		c: &hir::Constraint,
		data: &hir::ItemData,
		top_level: bool,
	) -> ConstraintId {
		let types = self.db.lookup_item_types(item);
		let mut collector = ExpressionCollector::new(self, data, item, &types);
		let mut constraint = Constraint::new(top_level, collector.collect_expression(c.expression));
		constraint.annotations_mut().extend(
			c.annotations
				.iter()
				.map(|ann| collector.collect_expression(*ann)),
		);
		self.model.add_constraint(Item::new(constraint, item))
	}

	/// Collect a declaration item
	pub fn collect_declaration(
		&mut self,
		item: ItemRef,
		d: &hir::Declaration,
		data: &hir::ItemData,
		top_level: bool,
	) -> Vec<DeclOrConstraint> {
		let types = self.db.lookup_item_types(item);

		let ty = match &types[d.pattern] {
			PatternTy::Variable(ty) => *ty,
			PatternTy::Destructuring(ty) => *ty,
			_ => unreachable!(),
		};
		let mut collector = ExpressionCollector::new(self, data, item, &types);
		let domain = collector.collect_domain(d.declared_type, ty, false);
		let mut decl = Declaration::new(top_level, domain);
		if data[d.declared_type].is_new(data) {
			if ty.is_set(collector.parent.db.upcast()) {
				if ty.inst(collector.parent.db.upcast()) == Some(VarType::Par) {
					match &**decl.domain() {
						DomainData::Set(boxed_domain, _) => match &***boxed_domain {
							DomainData::Bounded(e) => {
								decl.set_definition((**e).clone());
							}
							_ => unreachable!(),
						},
						_ => unreachable!(),
					}
				}
			} else if ty.opt(collector.parent.db.upcast()) == Some(OptType::Opt) {
				if ty.inst(collector.parent.db.upcast()) == Some(VarType::Par) {
					match &**decl.domain() {
						DomainData::Bounded(e) => {
							let card_e = alloc_expression(LookupCall {
								function: collector.parent.ids.card.into(),
								arguments: vec![(**e).clone()],
							}, &collector, item);
							let zero = alloc_expression(IntegerLiteral(0), &collector, item);
							let eq = alloc_expression(LookupCall {
								function: collector.parent.ids.eq.into(),
								arguments: vec![card_e, zero],
							}, &collector, item);
							let absent = alloc_expression(Absent, &collector, item);
							let min_e = alloc_expression(LookupCall {
								function: collector.parent.ids.min.into(),
								arguments: vec![(**e).clone()],
							}, &collector, item);
							let if_then_else = alloc_expression(IfThenElse {
								branches: vec![Branch {
									condition: eq,
									result: absent,
								}],
								else_result: Box::new(min_e),
							}, &collector, item);
							decl.set_definition(if_then_else);
						}
						_ => unreachable!(),
					}
				}
			} else {
				match &**decl.domain() {
					DomainData::Bounded(e) => match &***e {
						ExpressionData::SetLiteral(SetLiteral(element)) => {
							decl.set_definition(element[0].clone());
						}
						_ => unreachable!(),
					},
					_ => unreachable!(),
				}
			}
		} else if let Some(def) = d.definition {
			decl.set_definition(collector.collect_expression(def));
		}
		let idx = collector
			.parent
			.model
			.add_declaration(Item::new(decl, item));
		let decls = collector.collect_destructuring(idx, top_level, d.pattern);
		let mut ids = vec![idx.into()];
		collector.parent.model[idx]
			.annotations_mut()
			.reserve(d.annotations.len());
		for ann in d.annotations.iter().copied() {
			match collector.collect_declaration_annotation(idx, ann) {
				LoweredAnnotation::Expression(e) => {
					collector.parent.model[idx].annotations_mut().push(e)
				}
				LoweredAnnotation::Items(items) => ids.extend(items),
			}
		}
		ids.extend(decls.into_iter().map(DeclOrConstraint::Declaration));
		ids
	}

	/// Collect an enumeration item
	pub fn collect_enumeration(
		&mut self,
		item: ItemRef,
		e: &hir::Item<hir::Enumeration>,
	) -> EnumerationId {
		let types = self.db.lookup_item_types(item);
		let ty = &types[e.pattern];
		match ty {
			PatternTy::Enum(ty) => match ty.lookup(self.db.upcast()) {
				TyData::Set(VarType::Par, OptType::NonOpt, element) => {
					match element.lookup(self.db.upcast()) {
						TyData::Enum(_, _, t) => {
							let mut enumeration = Enumeration::new(t);
							{
								let mut collector =
									ExpressionCollector::new(self, &e.data, item, &types);
								enumeration.annotations_mut().extend(
									e.annotations
										.iter()
										.map(|ann| collector.collect_expression(*ann)),
								);
							}
							if let Some(def) = &e.definition {
								enumeration.set_definition(
									def.iter()
										.map(|c| self.collect_enum_case(c, &e.data, item, &types)),
								)
							}
							let idx = self.model.add_enumeration(Item::new(enumeration, item));
							self.resolutions.insert(
								PatternRef::new(item, e.pattern),
								LoweredIdentifier::ResolvedIdentifier(idx.into()),
							);
							self.add_enum_resolutions(
								idx,
								item,
								e.definition.iter().flat_map(|cs| cs.iter()),
							);
							idx
						}
						_ => unreachable!(),
					}
				}
				_ => unreachable!(),
			},
			_ => unreachable!(),
		}
	}

	/// Collect an enum assignment item
	pub fn collect_enumeration_assignment(
		&mut self,
		item: ItemRef,
		a: &hir::Item<hir::EnumAssignment>,
	) {
		let types = self.db.lookup_item_types(item);
		let res = types.name_resolution(a.assignee).unwrap();
		let idx = match &self.resolutions[&res] {
			LoweredIdentifier::ResolvedIdentifier(ResolvedIdentifier::Enumeration(e)) => *e,
			_ => unreachable!(),
		};
		let def = a
			.definition
			.iter()
			.map(|c| self.collect_enum_case(c, &a.data, item, &types))
			.collect::<Vec<_>>();
		self.model[idx].set_definition(def);
		self.add_enum_resolutions(idx, item, a.definition.iter());
	}

	fn add_enum_resolutions<'i>(
		&mut self,
		idx: EnumerationId,
		item: ItemRef,
		ecs: impl Iterator<Item = &'i hir::EnumConstructor>,
	) {
		for (i, ec) in ecs.enumerate() {
			match ec {
				hir::EnumConstructor::Named(hir::Constructor::Atom { pattern }) => {
					self.resolutions.insert(
						PatternRef::new(item, *pattern),
						LoweredIdentifier::ResolvedIdentifier(
							EnumMemberId::new(idx, i as u32).into(),
						),
					);
				}
				hir::EnumConstructor::Named(hir::Constructor::Function {
					constructor,
					destructor,
					..
				}) => {
					self.resolutions.insert(
						PatternRef::new(item, *constructor),
						LoweredIdentifier::Callable(Callable::EnumConstructor(EnumMemberId::new(
							idx, i as u32,
						))),
					);
					self.resolutions.insert(
						PatternRef::new(item, *destructor),
						LoweredIdentifier::Callable(Callable::EnumDestructor(EnumMemberId::new(
							idx, i as u32,
						))),
					);
				}
				_ => (),
			}
		}
	}

	fn collect_enum_case(
		&mut self,
		c: &hir::EnumConstructor,
		data: &hir::ItemData,
		item: ItemRef,
		types: &TypeResult,
	) -> Constructor {
		let (name, params) = match (c, &types[c.constructor_pattern()]) {
			(crate::hir::EnumConstructor::Named(crate::hir::Constructor::Atom { pattern }), _) => {
				return Constructor {
					name: data[*pattern].identifier(),
					parameters: None,
				}
			}
			(
				crate::hir::EnumConstructor::Named(crate::hir::Constructor::Function {
					constructor,
					parameters,
					..
				}),
				PatternTy::EnumConstructor(ecs),
			) => (
				data[*constructor].identifier(),
				ecs[0]
					.overload
					.params()
					.iter()
					.zip(parameters.iter())
					.map(|(ty, t)| (*ty, t.declared_type))
					.collect::<Vec<_>>(),
			),
			(
				crate::hir::EnumConstructor::Anonymous { parameters, .. },
				PatternTy::AnonymousEnumConstructor(f),
			) => (
				None,
				f.overload
					.params()
					.iter()
					.zip(parameters.iter())
					.map(|(ty, t)| (*ty, t.declared_type))
					.collect::<Vec<_>>(),
			),
			_ => unreachable!(),
		};

		Constructor {
			name,
			parameters: Some(
				params
					.iter()
					.map(|(ty, t)| {
						let mut collector = ExpressionCollector::new(self, data, item, types);
						let domain = collector.collect_domain(*t, *ty, false);
						let declaration = Declaration::new(false, domain);
						self.model.add_declaration(Item::new(declaration, item))
					})
					.collect(),
			),
		}
	}

	/// Collect a function item
	pub fn collect_function(&mut self, item: ItemRef, f: &hir::Item<hir::Function>) -> FunctionId {
		let types = self.db.lookup_item_types(item);
		let mut collector = ExpressionCollector::new(self, &f.data, item, &types);
		let res = PatternRef::new(item, f.pattern);
		match &types[f.pattern] {
			PatternTy::Function(fn_entry) => {
				let domain =
					collector.collect_domain(f.return_type, fn_entry.overload.return_type(), false);
				let mut function =
					Function::new(f.data[f.pattern].identifier().unwrap().into(), domain);
				function.annotations_mut().extend(
					f.annotations
						.iter()
						.map(|ann| collector.collect_expression(*ann)),
				);
				function.set_type_inst_vars(f.type_inst_vars.iter().map(|t| {
					match &types[t.name] {
						PatternTy::TyVar(tv) => tv.clone(),
						_ => unreachable!(),
					}
				}));

				let parameters = f
					.parameters
					.iter()
					.zip(fn_entry.overload.params())
					.map(|(param, ty)| {
						collector
							.parent
							.collect_fn_param(param, *ty, &f.data, item, &types)
					})
					.collect::<Vec<_>>();
				function.set_parameters(parameters);

				let idx = self.model.add_function(Item::new(function, item));
				self.resolutions
					.insert(res, LoweredIdentifier::Callable(Callable::Function(idx)));
				if f.body.is_some() {
					self.deferred.push((idx, item));
				}
				idx
			}
			_ => unreachable!(),
		}
	}

	fn collect_fn_param(
		&mut self,
		param: &crate::hir::Parameter,
		ty: Ty,
		data: &hir::ItemData,
		item: ItemRef,
		types: &TypeResult,
	) -> DeclarationId {
		let mut collector = ExpressionCollector::new(self, data, item, types);
		let domain = collector.collect_domain(param.declared_type, ty, false);
		let mut declaration = Declaration::new(false, domain);
		if let Some(p) = param.pattern.and_then(|p| data[p].identifier()) {
			declaration.set_name(p);
		}
		declaration.annotations_mut().extend(
			param
				.annotations
				.iter()
				.map(|ann| collector.collect_expression(*ann)),
		);
		self.model.add_declaration(Item::new(declaration, item))
	}

	/// Collect an output item
	pub fn collect_output(&mut self, item: ItemRef, o: &hir::Item<hir::Output>) -> OutputId {
		let types = self.db.lookup_item_types(item);
		let mut collector = ExpressionCollector::new(self, &o.data, item, &types);
		let mut output = Output::new(collector.collect_expression(o.expression));
		if let Some(s) = o.section {
			output.set_section(collector.collect_expression(s));
		}
		self.model.add_output(Item::new(output, item))
	}

	/// Collect solve item
	pub fn collect_solve(&mut self, item: ItemRef, s: &hir::Item<hir::Solve>) {
		let types = self.db.lookup_item_types(item);
		let mut optimise = |pattern: ArenaIndex<hir::Pattern>,
		                    objective: ArenaIndex<hir::Expression>,
		                    is_maximize: bool| match &types[pattern] {
			PatternTy::Variable(ty) => {
				let objective_origin = EntityRef::new(self.db.upcast(), item, objective);
				let mut collector = ExpressionCollector::new(self, &s.data, item, &types);
				let mut declaration = Declaration::new(
					true,
					Domain::unbounded(collector.parent.db, objective_origin, *ty),
				);
				if let Some(name) = s.data[pattern].identifier() {
					declaration.set_name(name);
				}
				let obj = collector.collect_expression(objective);
				declaration.set_definition(obj);
				let idx = self.model.add_declaration(Item::new(declaration, item));
				self.resolutions.insert(
					PatternRef::new(item, pattern),
					LoweredIdentifier::ResolvedIdentifier(idx.into()),
				);
				if is_maximize {
					Solve::maximize(idx)
				} else {
					Solve::minimize(idx)
				}
			}
			_ => unreachable!(),
		};
		let mut si = match &s.goal {
			hir::Goal::Maximize { pattern, objective } => optimise(*pattern, *objective, true),
			hir::Goal::Minimize { pattern, objective } => optimise(*pattern, *objective, false),
			hir::Goal::Satisfy => Solve::satisfy(),
		};
		let mut collector = ExpressionCollector::new(self, &s.data, item, &types);
		si.annotations_mut().extend(
			s.annotations
				.iter()
				.map(|ann| collector.collect_expression(*ann)),
		);
		self.model.set_solve(Item::new(si, item));
	}

	fn collect_type_alias(&mut self, item: ItemRef, ta: &hir::Item<hir::TypeAlias>) {
		let types = self.db.lookup_item_types(item);
		for e in hir::Type::expressions(ta.aliased_type, &ta.data) {
			if let Some(res) = types.name_resolution(e) {
				let res_types = self.db.lookup_item_types(res.item());
				if matches!(&res_types[res.pattern()], PatternTy::TypeAlias { .. }) {
					// Skip type aliases inside other type aliases (already will be processed)
					continue;
				}
			}
			// Create a declaration with the value of each expression used in a type alias
			let expression =
				ExpressionCollector::new(self, &ta.data, item, &types).collect_expression(e);
			let decl = Declaration::from_expression(self.db, true, expression);
			let idx = self
				.model
				.add_declaration(Item::new(decl, EntityRef::new(self.db.upcast(), item, e)));
			self.type_alias_expressions
				.insert(ExpressionRef::new(item, e), idx);
		}
	}

	/// Collect deferred function bodies
	pub fn collect_deferred(&mut self) {
		for (func, item) in self.deferred.clone().into_iter() {
			let types = self.db.lookup_item_types(item);
			let model = item.model(self.db.upcast());
			let local_item = item.local_item_ref(self.db.upcast());
			match local_item {
				LocalItemRef::Function(f) => {
					let mut function = self.model[func].clone();
					let param_decls = function.parameters().to_owned();
					let mut decls = Vec::new();
					let mut collector =
						ExpressionCollector::new(self, &model[f].data, item, &types);
					for (decl, param) in param_decls.into_iter().zip(model[f].parameters.iter()) {
						if let Some(p) = param.pattern {
							let dsts = collector.collect_destructuring(decl, false, p);
							decls.extend(dsts);
						}
					}
					let body = model[f].body.unwrap();
					let collected_body = collector.collect_expression(body);
					let e = if decls.is_empty() {
						collected_body
					} else {
						let origin = EntityRef::new(collector.parent.db.upcast(), item, body);
						Expression::new(
							self.db,
							&self.model,
							origin,
							Let {
								items: decls.into_iter().map(LetItem::Declaration).collect(),
								in_expression: Box::new(collected_body),
							},
						)
					};
					function.set_body(e);
					self.model[func] = function;
				}
				_ => unreachable!(),
			}
		}
	}

	/// Finish lowering
	pub fn finish(self) -> Model {
		self.model
	}
}

struct ExpressionCollector<'a, 'b> {
	parent: &'a mut ItemCollector<'b>,
	data: &'a hir::ItemData,
	item: ItemRef,
	types: &'a TypeResult,
}

impl<'a, 'b> ExpressionCollector<'a, 'b> {
	fn new(
		parent: &'a mut ItemCollector<'b>,
		data: &'a crate::hir::ItemData,
		item: ItemRef,
		types: &'a TypeResult,
	) -> Self {
		Self {
			parent,
			data,
			item,
			types,
		}
	}

	fn introduce_declaration(
		&mut self,
		top_level: bool,
		origin: impl Into<Origin>,
		f: impl FnOnce(&mut ExpressionCollector<'_, '_>) -> Expression,
	) -> DeclarationId {
		let origin: Origin = origin.into();
		let mut collector = ExpressionCollector::new(self.parent, self.data, self.item, self.types);
		let def = f(&mut collector);
		let decl = Declaration::from_expression(self.parent.db, top_level, def);
		self.parent.model.add_declaration(Item::new(decl, origin))
	}

	/// Collect an expression
	pub fn collect_expression(&mut self, idx: ArenaIndex<hir::Expression>) -> Expression {
		maybe_grow_stack(|| self.collect_expression_inner(idx))
	}

	pub fn collect_expression_inner(&mut self, idx: ArenaIndex<hir::Expression>) -> Expression {
		let db = self.parent.db;
		let ty = self.types[idx];
		let origin = EntityRef::new(db.upcast(), self.item, idx);
		let mut result = match &self.data[idx] {
			hir::Expression::Absent => alloc_expression(Absent, self, origin),
			hir::Expression::ArrayAccess(aa) => {
				let is_slice = match self.types[aa.indices].lookup(db.upcast()) {
					TyData::Tuple(_, fs) => fs.iter().any(|f| f.is_set(db.upcast())),
					TyData::Set(_, _, _) => true,
					_ => false,
				};
				if is_slice {
					self.collect_slice(aa.collection, aa.indices, origin)
				} else {
					let c = self.collect_expression(aa.collection);
					let i = self.collect_expression(aa.indices);
					self.collect_array_access(c, i, origin)
				}
			}
			hir::Expression::ArrayComprehension(c) => {
				let mut generators = Vec::with_capacity(c.generators.len());
				for g in c.generators.iter() {
					self.collect_generator(g, &mut generators);
				}
				alloc_expression(
					ArrayComprehension {
						generators,
						template: Box::new(self.collect_expression(c.template)),
						indices: c
							.indices
							.map(|indices| Box::new(self.collect_expression(indices))),
					},
					self,
					origin,
				)
			}
			hir::Expression::ArrayLiteral(al) => alloc_expression(
				ArrayLiteral(
					al.members
						.iter()
						.map(|m| self.collect_expression(*m))
						.collect(),
				),
				self,
				origin,
			),
			// Desugar 2D array literal into array2d call
			hir::Expression::ArrayLiteral2D(al) => {
				let mut idx_array = |dim: &hir::MaybeIndexSet| match dim {
					hir::MaybeIndexSet::Indexed(es) => alloc_expression(
						ArrayLiteral(es.iter().map(|e| self.collect_expression(*e)).collect()),
						self,
						origin,
					),
					hir::MaybeIndexSet::NonIndexed(c) => alloc_expression(
						LookupCall {
							function: self.parent.ids.set2array.into(),
							arguments: vec![if *c > 0 {
								alloc_expression(
									LookupCall {
										function: self.parent.ids.dot_dot.into(),
										arguments: vec![
											alloc_expression(IntegerLiteral(1), self, origin),
											alloc_expression(
												IntegerLiteral(*c as i64),
												self,
												origin,
											),
										],
									},
									self,
									origin,
								)
							} else {
								alloc_expression(SetLiteral(Vec::new()), self, origin)
							}],
						},
						self,
						origin,
					),
				};
				let rows = idx_array(&al.rows);
				let columns = idx_array(&al.columns);
				alloc_expression(
					LookupCall {
						function: self.parent.ids.array2d.into(),
						arguments: vec![
							rows,
							columns,
							alloc_expression(
								ArrayLiteral(
									al.members
										.iter()
										.map(|e| self.collect_expression(*e))
										.collect(),
								),
								self,
								origin,
							),
						],
					},
					self,
					origin,
				)
			}
			// Desugar indexed array literal into arrayNd call
			hir::Expression::IndexedArrayLiteral(al) => alloc_expression(
				LookupCall {
					function: self.parent.ids.array_nd.into(),
					arguments: vec![
						if al.indices.len() == 1 {
							self.collect_expression(al.indices[0])
						} else {
							alloc_expression(
								ArrayLiteral(
									al.indices
										.iter()
										.map(|e| self.collect_expression(*e))
										.collect(),
								),
								self,
								origin,
							)
						},
						alloc_expression(
							ArrayLiteral(
								al.members
									.iter()
									.map(|e| self.collect_expression(*e))
									.collect(),
							),
							self,
							origin,
						),
					],
				},
				self,
				origin,
			),
			hir::Expression::BooleanLiteral(b) => alloc_expression(*b, self, origin),
			hir::Expression::Call(c) => {
				let function = if let hir::Expression::Identifier(_) = self.data[c.function] {
					let res = self.types.name_resolution(c.function).unwrap();
					let ident = self.parent.resolutions.get(&res).unwrap_or_else(|| {
						panic!(
							"Did not lower {:?} at {:?} used by {:?} at {:?}",
							res,
							NodeRef::from(res.into_entity(self.parent.db.upcast()))
								.source_span(self.parent.db.upcast()),
							ExpressionRef::new(self.item, c.function),
							NodeRef::from(EntityRef::new(
								self.parent.db.upcast(),
								self.item,
								c.function
							))
							.source_span(self.parent.db.upcast()),
						)
					});
					match ident {
						LoweredIdentifier::Callable(c) => c.clone(),
						_ => Callable::Expression(Box::new(self.collect_expression(c.function))),
					}
				} else {
					Callable::Expression(Box::new(self.collect_expression(c.function)))
				};
				alloc_expression(
					Call {
						function,
						arguments: c
							.arguments
							.iter()
							.map(|arg| self.collect_expression(*arg))
							.collect(),
					},
					self,
					origin,
				)
			}
			hir::Expression::Case(c) => {
				let scrutinee_origin =
					EntityRef::new(self.parent.db.upcast(), self.item, c.expression);
				let scrutinee = self.introduce_declaration(false, scrutinee_origin, |collector| {
					collector.collect_expression(c.expression)
				});
				alloc_expression(
					Let {
						items: vec![LetItem::Declaration(scrutinee)],
						in_expression: Box::new(alloc_expression(
							Case {
								scrutinee: Box::new(alloc_expression(scrutinee, self, origin)),
								branches: c
									.cases
									.iter()
									.map(|case| {
										let pattern_origin = EntityRef::new(
											self.parent.db.upcast(),
											self.item,
											case.pattern,
										);
										let pattern = self.collect_pattern(case.pattern);
										let decls = self.collect_destructuring(
											scrutinee,
											false,
											case.pattern,
										);
										let result = self.collect_expression(case.value);
										if decls.is_empty() {
											CaseBranch::new(pattern, result)
										} else {
											CaseBranch::new(
												pattern,
												alloc_expression(
													Let {
														items: decls
															.into_iter()
															.map(LetItem::Declaration)
															.collect(),
														in_expression: Box::new(result),
													},
													self,
													pattern_origin,
												),
											)
										}
									})
									.collect(),
							},
							self,
							origin,
						)),
					},
					self,
					origin,
				)
			}
			hir::Expression::FloatLiteral(f) => alloc_expression(*f, self, origin),
			hir::Expression::Identifier(_) => {
				let res = self.types.name_resolution(idx).unwrap();
				let ident = self.parent.resolutions.get(&res).unwrap_or_else(|| {
					panic!(
						"Did not lower {:?} at {:?} used by {:?} at {:?}",
						res,
						NodeRef::from(res.into_entity(self.parent.db.upcast()))
							.source_span(self.parent.db.upcast()),
						ExpressionRef::new(self.item, idx),
						NodeRef::from(EntityRef::new(self.parent.db.upcast(), self.item, idx))
							.source_span(self.parent.db.upcast()),
					)
				});
				let expr = alloc_expression(
					match ident {
						LoweredIdentifier::ResolvedIdentifier(i) => i.clone(),
						_ => unreachable!(),
					},
					self,
					origin,
				);

				if expr.ty() != ty && expr.ty().make_par(db.upcast()) == ty {
					// Need to insert call to fix()
					assert_eq!(expr.ty().make_par(db.upcast()), ty);
					alloc_expression(
						LookupCall {
							function: self.parent.ids.fix.into(),
							arguments: vec![expr],
						},
						self,
						origin,
					)
				} else {
					expr
				}
			}
			hir::Expression::IfThenElse(ite) => alloc_expression(
				IfThenElse {
					branches: ite
						.branches
						.iter()
						.map(|b| {
							Branch::new(
								self.collect_expression(b.condition),
								self.collect_expression(b.result),
							)
						})
						.collect(),
					else_result: Box::new(
						ite.else_result
							.map(|e| self.collect_expression(e))
							.unwrap_or_else(|| self.collect_default_else(ty, origin.into())),
					),
				},
				self,
				origin,
			),
			hir::Expression::Infinity => alloc_expression(Infinity, self, origin),
			hir::Expression::IntegerLiteral(i) => alloc_expression(*i, self, origin),
			hir::Expression::Lambda(l) => {
				let fn_type = match ty.lookup(db.upcast()) {
					TyData::Function(_, f) => f,
					_ => unreachable!(),
				};
				let return_type = l
					.return_type
					.map(|r| self.collect_domain(r, fn_type.return_type, false))
					.unwrap_or_else(|| {
						Domain::unbounded(self.parent.db, origin, fn_type.return_type)
					});
				let mut decls = Vec::new();
				let parameters = l
					.parameters
					.iter()
					.zip(fn_type.params.iter())
					.map(|(param, ty)| {
						let decl = self
							.parent
							.collect_fn_param(param, *ty, self.data, self.item, self.types);
						if let Some(p) = param.pattern {
							decls.extend(self.collect_destructuring(decl, false, p));
						}
						decl
					})
					.collect::<Vec<_>>();
				let body = self.collect_expression(l.body);
				let function = Function::lambda(
					return_type,
					parameters,
					if decls.is_empty() {
						body
					} else {
						let body_entity =
							EntityRef::new(self.parent.db.upcast(), self.item, l.body);
						alloc_expression(
							Let {
								items: decls.into_iter().map(LetItem::Declaration).collect(),
								in_expression: Box::new(body),
							},
							self,
							body_entity,
						)
					},
				);
				let f = self.parent.model.add_function(Item::new(function, origin));
				alloc_expression(Lambda(f), self, origin)
			}
			hir::Expression::Let(l) => alloc_expression(
				Let {
					items: l
						.items
						.iter()
						.flat_map(|i| match i {
							hir::LetItem::Constraint(c) => {
								let constraint = self
									.parent
									.collect_constraint(self.item, c, self.data, false);
								vec![LetItem::Constraint(constraint)]
							}
							hir::LetItem::Declaration(d) => self
								.parent
								.collect_declaration(self.item, d, self.data, false)
								.into_iter()
								.map(|d| d.into())
								.collect::<Vec<_>>(),
						})
						.collect(),
					in_expression: Box::new(self.collect_expression(l.in_expression)),
				},
				self,
				origin,
			),
			hir::Expression::RecordAccess(ra) => {
				let record = self.collect_expression(ra.record);
				if self.types[ra.record].is_class(self.parent.db.upcast()) {
					let class_type = self.types[ra.record]
						.class_type(self.parent.db.upcast())
						.unwrap();

					let mut access_expr = record;

					let class_pattern_ref = class_type.pattern(self.parent.db.upcast());
					let class_objects_infos = &self.parent.class_objects[&class_pattern_ref];

					for (super_class_idx, super_class) in
						class_type.superclasses(self.parent.db.upcast()).enumerate()
					{
						if let Some(super_class_constr_idx) =
							class_objects_infos[super_class_idx].class_enum_member
						{
							let super_enum = match self.parent.resolutions
								[&super_class.pattern(self.parent.db.upcast())]
							{
								LoweredIdentifier::ResolvedIdentifier(
									ResolvedIdentifier::Enumeration(i),
								) => i,
								_ => unreachable!(),
							};
							let enum_member = EnumMemberId::new(super_enum, super_class_constr_idx);

							access_expr = alloc_expression(
								Call {
									function: Callable::EnumConstructor(enum_member),
									arguments: vec![access_expr],
								},
								self,
								origin,
							);
						}

						let have_attr = super_class
							.attributes
							.iter()
							.any(|(attr_id, _)| *attr_id == ra.field);
						if have_attr {
							let superclass_pattern_ref =
								super_class.pattern(self.parent.db.upcast());
							let objects_decl =
								&self.parent.class_objects[&superclass_pattern_ref][0];
							let objects_decl_id =
								alloc_expression(objects_decl.class_objects, self, origin);

							let erase_enum = alloc_expression(
								LookupCall {
									function: self.parent.ids.erase_enum.into(),
									arguments: vec![access_expr],
								},
								self,
								origin,
							);
							let array_access = alloc_expression(
								LookupCall {
									function: self.parent.ids.array_access.into(),
									arguments: vec![objects_decl_id, erase_enum],
								},
								self,
								origin,
							);
							return alloc_expression(
								RecordAccess {
									record: Box::new(array_access),
									field: ra.field,
								},
								self,
								origin,
							);
						}
					}
					unreachable!()
				} else if self.types[ra.record].is_array(self.parent.db.upcast()) {
					// Lift to comprehension
					let record_ty = record.ty().elem_ty(self.parent.db.upcast()).unwrap();
					let declaration = Declaration::new(
						false,
						Domain::unbounded(self.parent.db, origin, record_ty),
					);
					let idx = self
						.parent
						.model
						.add_declaration(Item::new(declaration, origin));
					let g = Generator::Iterator {
						declarations: vec![idx],
						collection: record,
						where_clause: None,
					};
					alloc_expression(
						ArrayComprehension {
							generators: vec![g],
							template: Box::new(alloc_expression(
								RecordAccess {
									record: Box::new(alloc_expression(idx, self, origin)),
									field: ra.field,
								},
								self,
								origin,
							)),
							indices: None,
						},
						self,
						origin,
					)
				} else {
					alloc_expression(
						RecordAccess {
							record: Box::new(record),
							field: ra.field,
						},
						self,
						origin,
					)
				}
			}
			hir::Expression::RecordLiteral(rl) => alloc_expression(
				RecordLiteral(
					rl.fields
						.iter()
						.map(|(i, v)| {
							(
								self.data[*i].identifier().unwrap(),
								self.collect_expression(*v),
							)
						})
						.collect(),
				),
				self,
				origin,
			),
			hir::Expression::SetComprehension(c) => {
				let mut generators = Vec::with_capacity(c.generators.len());
				for g in c.generators.iter() {
					self.collect_generator(g, &mut generators);
				}
				alloc_expression(
					SetComprehension {
						generators,
						template: Box::new(self.collect_expression(c.template)),
					},
					self,
					origin,
				)
			}
			hir::Expression::SetLiteral(sl) => alloc_expression(
				SetLiteral(
					sl.members
						.iter()
						.map(|m| self.collect_expression(*m))
						.collect(),
				),
				self,
				origin,
			),
			hir::Expression::Slice(_) => {
				unreachable!("Slice used outside of array access")
			}
			hir::Expression::StringLiteral(sl) => alloc_expression(sl.clone(), self, origin),
			hir::Expression::TupleAccess(ta) => {
				let tuple = self.collect_expression(ta.tuple);
				if self.types[ta.tuple].is_array(self.parent.db.upcast()) {
					// Lift to comprehension
					let tuple_ty = tuple.ty().elem_ty(self.parent.db.upcast()).unwrap();
					let declaration = Declaration::new(
						false,
						Domain::unbounded(self.parent.db, origin, tuple_ty),
					);
					let idx = self
						.parent
						.model
						.add_declaration(Item::new(declaration, origin));
					let g = Generator::Iterator {
						declarations: vec![idx],
						collection: tuple,
						where_clause: None,
					};
					alloc_expression(
						ArrayComprehension {
							generators: vec![g],
							template: Box::new(alloc_expression(
								TupleAccess {
									tuple: Box::new(alloc_expression(idx, self, origin)),
									field: ta.field,
								},
								self,
								origin,
							)),
							indices: None,
						},
						self,
						origin,
					)
				} else {
					alloc_expression(
						TupleAccess {
							tuple: Box::new(tuple),
							field: ta.field,
						},
						self,
						origin,
					)
				}
			}
			hir::Expression::TupleLiteral(tl) => alloc_expression(
				TupleLiteral(
					tl.fields
						.iter()
						.map(|f| self.collect_expression(*f))
						.collect(),
				),
				self,
				origin,
			),
			hir::Expression::Missing => unreachable!("Missing expression"),
		};
		result.annotations_mut().extend(
			self.data
				.annotations(idx)
				.map(|ann| self.collect_expression(ann)),
		);
		// assert_eq!(
		// 	result.ty(),
		// 	ty,
		// 	"Type by construction ({}) disagrees with typechecker ({}) at {:?}",
		// 	result.ty().pretty_print(db.upcast()),
		// 	ty.pretty_print(db.upcast()),
		// 	NodeRef::from(origin).source_span(db.upcast())
		// );
		result
	}

	fn collect_declaration_annotation(
		&mut self,
		decl: DeclarationId,
		ann: ArenaIndex<hir::Expression>,
	) -> LoweredAnnotation {
		// Declarations can have annotations which point to functions using ::annotated_expression.
		// These need to be desugared into constraints.
		match &self.data[ann] {
			hir::Expression::Identifier(_) => {
				let res = self.types.name_resolution(ann).unwrap();
				let ident = self.parent.resolutions.get(&res).unwrap_or_else(|| {
					panic!(
						"Did not lower {:?} at {:?} used by {:?} at {:?}",
						res,
						NodeRef::from(res.into_entity(self.parent.db.upcast()))
							.source_span(self.parent.db.upcast()),
						ExpressionRef::new(self.item, ann),
						NodeRef::from(EntityRef::new(self.parent.db.upcast(), self.item, ann))
							.source_span(self.parent.db.upcast()),
					)
				});
				if let LoweredIdentifier::Callable(function) = ident.clone() {
					let origin = EntityRef::new(self.parent.db.upcast(), self.item, ann);
					let ann_decl = self.introduce_declaration(
						self.parent.model[decl].top_level(),
						origin,
						|collector| {
							// Call annotation function using the annotated declaration
							let arguments = vec![alloc_expression(
								ResolvedIdentifier::Declaration(decl),
								collector,
								origin,
							)];
							alloc_expression(
								Call {
									function: function.clone(),
									arguments,
								},
								collector,
								origin,
							)
						},
					);

					let annotate = alloc_expression(
						LookupCall {
							function: self.parent.ids.annotate.into(),
							arguments: vec![
								alloc_expression(
									ResolvedIdentifier::Declaration(decl),
									self,
									origin,
								),
								alloc_expression(
									ResolvedIdentifier::Declaration(ann_decl),
									self,
									origin,
								),
							],
						},
						self,
						origin,
					);
					let constraint = Constraint::new(self.parent.model[decl].top_level(), annotate);
					let c_idx = self
						.parent
						.model
						.add_constraint(Item::new(constraint, origin));

					return LoweredAnnotation::Items(vec![ann_decl.into(), c_idx.into()]);
				}
			}
			hir::Expression::Call(c) => {
				let origin = EntityRef::new(self.parent.db.upcast(), self.item, ann);
				let function = if let hir::Expression::Identifier(_) = self.data[c.function] {
					let res = self.types.name_resolution(c.function).unwrap();
					let ident = self.parent.resolutions.get(&res).unwrap_or_else(|| {
						panic!(
							"Did not lower {:?} at {:?} used by {:?} at {:?}",
							res,
							NodeRef::from(res.into_entity(self.parent.db.upcast()))
								.source_span(self.parent.db.upcast()),
							ExpressionRef::new(self.item, c.function),
							NodeRef::from(EntityRef::new(
								self.parent.db.upcast(),
								self.item,
								c.function
							))
							.source_span(self.parent.db.upcast()),
						)
					});
					match ident {
						LoweredIdentifier::Callable(c) => c.clone(),
						_ => Callable::Expression(Box::new(self.collect_expression(c.function))),
					}
				} else {
					Callable::Expression(Box::new(self.collect_expression(c.function)))
				};

				if let Callable::Function(f) = &function {
					if self.parent.model[*f].parameters().len() > c.arguments.len() {
						// Add the annotated declaration identifier as first argument
						let mut arguments = Vec::with_capacity(c.arguments.len() + 1);
						arguments.push(alloc_expression(
							ResolvedIdentifier::Declaration(decl),
							self,
							origin,
						));
						arguments
							.extend(c.arguments.iter().map(|arg| self.collect_expression(*arg)));

						let ann_decl = self.introduce_declaration(
							self.parent.model[decl].top_level(),
							origin,
							|collector| {
								alloc_expression(
									Call {
										function,
										arguments,
									},
									collector,
									origin,
								)
							},
						);

						let annotate = alloc_expression(
							LookupCall {
								function: self.parent.ids.annotate.into(),
								arguments: vec![
									alloc_expression(
										ResolvedIdentifier::Declaration(decl),
										self,
										origin,
									),
									alloc_expression(
										ResolvedIdentifier::Declaration(ann_decl),
										self,
										origin,
									),
								],
							},
							self,
							origin,
						);
						let constraint =
							Constraint::new(self.parent.model[decl].top_level(), annotate);
						let c_idx = self
							.parent
							.model
							.add_constraint(Item::new(constraint, origin));

						return LoweredAnnotation::Items(vec![ann_decl.into(), c_idx.into()]);
					}
				}

				// Return as is
				return LoweredAnnotation::Expression(alloc_expression(
					Call {
						function,
						arguments: c
							.arguments
							.iter()
							.map(|arg| self.collect_expression(*arg))
							.collect(),
					},
					self,
					origin,
				));
			}
			_ => (),
		}
		LoweredAnnotation::Expression(self.collect_expression(ann))
	}

	/// Rewrite index slicing into a call
	///
	/// Turns all indices into sets to match the slicing builtin function, and then coerces to the correct output index set.
	fn collect_slice(
		&mut self,
		collection: ArenaIndex<hir::Expression>,
		indices: ArenaIndex<hir::Expression>,
		origin: impl Into<Origin>,
	) -> Expression {
		let origin: Origin = origin.into();
		let collection_entity = EntityRef::new(self.parent.db.upcast(), self.item, collection);
		let indices_entity = EntityRef::new(self.parent.db.upcast(), self.item, indices);

		let mut decls = Vec::new();
		let collection_decl = if matches!(&self.data[collection], hir::Expression::Identifier(_)) {
			let expr = self.collect_expression(collection);
			match &*expr {
				ExpressionData::Identifier(ResolvedIdentifier::Declaration(decl)) => *decl,
				_ => unreachable!(),
			}
		} else {
			// Add declaration to store collection
			let origin = EntityRef::new(self.parent.db.upcast(), self.item, collection);
			let decl = self.introduce_declaration(false, origin, |collector| {
				collector.collect_expression(collection)
			});
			decls.push(decl);
			decl
		};
		let mut index_sets_for_infinite_slice = None;
		let array_dims = self.types[collection]
			.dims(self.parent.db.upcast())
			.unwrap();
		let mut slices = Vec::with_capacity(array_dims);
		match self.types[indices].lookup(self.parent.db.upcast()) {
			TyData::Tuple(_, fs) => {
				if let hir::Expression::TupleLiteral(tl) = &self.data[indices] {
					for (i, (ty, e)) in fs.iter().zip(tl.fields.iter()).enumerate() {
						let index_entity = EntityRef::new(self.parent.db.upcast(), self.item, *e);
						let mut is_set = true;
						let decl = self.introduce_declaration(false, index_entity, |collector| {
							if let hir::Expression::Slice(s) = &collector.data[*e] {
								// Rewrite infinite slice .. into `'..'(index_set_mofn(c))`
								if index_sets_for_infinite_slice.is_none() {
									let decl = collector.introduce_declaration(
										false,
										origin,
										|collector| {
											alloc_expression(
												LookupCall {
													function: self.parent.ids.index_sets.into(),
													arguments: vec![alloc_expression(
														collection_decl,
														collector,
														collection_entity,
													)],
												},
												collector,
												origin,
											)
										},
									);
									decls.push(decl);
									index_sets_for_infinite_slice = Some(decl);
								}
								alloc_expression(
									LookupCall {
										function: (*s).into(),
										arguments: vec![alloc_expression(
											TupleAccess {
												tuple: Box::new(alloc_expression(
													index_sets_for_infinite_slice.unwrap(),
													collector,
													index_entity,
												)),
												field: IntegerLiteral(i as i64 + 1),
											},
											collector,
											index_entity,
										)],
									},
									collector,
									index_entity,
								)
							} else if ty.is_set(collector.parent.db.upcast()) {
								// Slice
								collector.collect_expression(*e)
							} else {
								// Rewrite index as slice of {i}
								is_set = false;
								alloc_expression(
									SetLiteral(vec![collector.collect_expression(*e)]),
									collector,
									index_entity,
								)
							}
						});
						slices.push((decl, is_set, index_entity));
						decls.push(decl);
					}
				} else {
					// Expression which evaluates to a tuple
					let indices_decl =
						self.introduce_declaration(false, indices_entity, |collector| {
							collector.collect_expression(indices)
						});
					decls.push(indices_decl);
					for (i, f) in fs.iter().enumerate() {
						// Create declaration for each index
						let is_set = f.is_set(self.parent.db.upcast());
						let accessor =
							self.introduce_declaration(false, indices_entity, |collector| {
								let ta = alloc_expression(
									TupleAccess {
										tuple: Box::new(alloc_expression(
											indices_decl,
											collector,
											indices_entity,
										)),
										field: IntegerLiteral(i as i64 + 1),
									},
									collector,
									indices_entity,
								);
								if is_set {
									ta
								} else {
									// Rewrite as {i}
									alloc_expression(
										SetLiteral(vec![ta]),
										collector,
										indices_entity,
									)
								}
							});

						slices.push((accessor, is_set, indices_entity));
						decls.push(accessor);
					}
				}
			}
			_ => {
				// 1D slicing, so must be a set index
				let decl = self.introduce_declaration(false, indices_entity, |collector| {
					if let hir::Expression::Slice(s) = &collector.data[indices] {
						// Rewrite infinite slice .. into `'..'(index_set(c))`
						alloc_expression(
							LookupCall {
								function: (*s).into(),
								arguments: vec![alloc_expression(
									LookupCall {
										function: collector.parent.ids.index_set.into(),
										arguments: vec![alloc_expression(
											collection_decl,
											collector,
											collection_entity,
										)],
									},
									collector,
									indices_entity,
								)],
							},
							collector,
							indices_entity,
						)
					} else {
						collector.collect_expression(indices)
					}
				});
				slices.push((decl, true, indices_entity));
				decls.push(decl);
			}
		}
		let collection_ident = alloc_expression(collection_decl, self, collection_entity);
		let slice_tuple = alloc_expression(
			TupleLiteral(
				slices
					.iter()
					.map(|(decl, _, origin)| alloc_expression(*decl, self, *origin))
					.collect(),
			),
			self,
			indices_entity,
		);
		let arguments = slices
			.iter()
			.filter_map(|(decl, is_slice, origin)| {
				if *is_slice {
					Some(alloc_expression(*decl, self, *origin))
				} else {
					None
				}
			})
			.chain([alloc_expression(
				LookupCall {
					function: self.parent.ids.array_access.into(),
					arguments: vec![collection_ident, slice_tuple],
				},
				self,
				origin,
			)])
			.collect::<Vec<_>>();
		alloc_expression(
			Let {
				items: decls.into_iter().map(LetItem::Declaration).collect(),
				in_expression: Box::new(alloc_expression(
					LookupCall {
						function: Identifier::new(
							format!("array{}d", arguments.len() - 1),
							self.parent.db.upcast(),
						)
						.into(),
						arguments,
					},
					self,
					origin,
				)),
			},
			self,
			origin,
		)
	}

	fn collect_array_access(
		&mut self,
		collection: Expression,
		indices: Expression,
		origin: impl Into<Origin>,
	) -> Expression {
		maybe_grow_stack(|| {
			let origin = origin.into();
			if indices.ty().contains_var(self.parent.db.upcast()) {
				let elem = collection.ty().elem_ty(self.parent.db.upcast()).unwrap();
				if elem.is_tuple(self.parent.db.upcast()) || elem.is_record(self.parent.db.upcast())
				{
					// Decompose access to array of structured types
					let c_origin = collection.origin();
					let c_idx = self.introduce_declaration(false, c_origin, |_| collection);
					let c_ident = alloc_expression(c_idx, self, c_origin);
					let i_origin = indices.origin();
					let i_idx = self.introduce_declaration(false, c_origin, |_| indices);
					let i_ident = alloc_expression(i_idx, self, i_origin);

					if let Some(fields) = elem.record_fields(self.parent.db.upcast()) {
						let mut decomposed = Vec::with_capacity(fields.len());
						for (k, _) in fields {
							let field = Identifier::from(k);
							let decl = Declaration::new(
								false,
								Domain::unbounded(self.parent.db, c_origin, elem),
							);
							let decl_idx =
								self.parent.model.add_declaration(Item::new(decl, c_origin));
							let generators = vec![Generator::Iterator {
								declarations: vec![decl_idx],
								collection: c_ident.clone(),
								where_clause: None,
							}];
							let comprehension = alloc_expression(
								ArrayComprehension {
									generators,
									indices: None,
									template: Box::new(alloc_expression(
										RecordAccess {
											record: Box::new(alloc_expression(
												decl_idx, self, c_origin,
											)),
											field,
										},
										self,
										origin,
									)),
								},
								self,
								origin,
							);
							let array = alloc_expression(
								LookupCall {
									function: self.parent.ids.array_xd.into(),
									arguments: vec![c_ident.clone(), comprehension],
								},
								self,
								origin,
							);
							decomposed.push((
								field,
								self.collect_array_access(array, i_ident.clone(), origin),
							));
						}
						return alloc_expression(
							Let {
								items: vec![
									LetItem::Declaration(c_idx),
									LetItem::Declaration(i_idx),
								],
								in_expression: Box::new(alloc_expression(
									RecordLiteral(decomposed),
									self,
									origin,
								)),
							},
							self,
							origin,
						);
					}
					let fields = elem.field_len(self.parent.db.upcast()).unwrap();
					let mut decomposed = Vec::with_capacity(fields);
					for i in 1..=(fields as i64) {
						let decl = Declaration::new(
							false,
							Domain::unbounded(self.parent.db, c_origin, elem),
						);
						let decl_idx = self.parent.model.add_declaration(Item::new(decl, c_origin));
						let generators = vec![Generator::Iterator {
							declarations: vec![decl_idx],
							collection: c_ident.clone(),
							where_clause: None,
						}];
						let comprehension = alloc_expression(
							ArrayComprehension {
								generators,
								indices: None,
								template: Box::new(alloc_expression(
									TupleAccess {
										tuple: Box::new(alloc_expression(decl_idx, self, c_origin)),
										field: IntegerLiteral(i),
									},
									self,
									origin,
								)),
							},
							self,
							origin,
						);
						let array = alloc_expression(
							LookupCall {
								function: self.parent.ids.array_xd.into(),
								arguments: vec![c_ident.clone(), comprehension],
							},
							self,
							origin,
						);
						decomposed.push(self.collect_array_access(array, i_ident.clone(), origin));
					}
					return alloc_expression(
						Let {
							items: vec![LetItem::Declaration(c_idx), LetItem::Declaration(i_idx)],
							in_expression: Box::new(alloc_expression(
								TupleLiteral(decomposed),
								self,
								origin,
							)),
						},
						self,
						origin,
					);
				}
			}
			alloc_expression(
				LookupCall {
					function: self.parent.ids.array_access.into(),
					arguments: vec![collection, indices],
				},
				self,
				origin,
			)
		})
	}

	fn collect_generator(&mut self, generator: &hir::Generator, generators: &mut Vec<Generator>) {
		let pattern_to_where = |c: &mut Self,
		                        decl: DeclarationId,
		                        p: ArenaIndex<hir::Pattern>,
		                        origin: Origin| {
			// Turn destructuring into where clause of case matching pattern
			let pattern = c.collect_pattern(p);
			alloc_expression(
				Case {
					scrutinee: Box::new(alloc_expression(decl, c, origin)),
					branches: vec![
						CaseBranch::new(pattern, alloc_expression(BooleanLiteral(true), c, origin)),
						CaseBranch::new(
							Pattern::anonymous(
								match &c.types[p] {
									PatternTy::Destructuring(ty) => *ty,
									_ => unreachable!(),
								},
								origin,
							),
							alloc_expression(BooleanLiteral(false), c, origin),
						),
					],
				},
				c,
				origin,
			)
		};

		match generator {
			hir::Generator::Iterator {
				patterns,
				collection,
				where_clause,
			} => {
				let mut assignments = Vec::new();
				let mut where_clauses = Vec::new();
				let declarations = patterns
					.iter()
					.map(|p| {
						let origin = EntityRef::new(self.parent.db.upcast(), self.item, *p);
						let ty = match &self.types[*p] {
							PatternTy::Variable(ty) | PatternTy::Destructuring(ty) => *ty,
							_ => unreachable!(),
						};
						let declaration =
							Declaration::new(false, Domain::unbounded(self.parent.db, origin, ty));
						let decl = self
							.parent
							.model
							.add_declaration(Item::new(declaration, origin));
						let asgs = self.collect_destructuring(decl, false, *p);
						if !asgs.is_empty() && hir::Pattern::is_refutable(*p, self.data) {
							where_clauses.push(pattern_to_where(self, decl, *p, origin.into()));
						}
						assignments.extend(asgs);
						decl
					})
					.collect();
				let collection = self.collect_expression(*collection);
				let where_clause = where_clause.map(|w| self.collect_expression(w));
				if assignments.is_empty() {
					generators.push(Generator::Iterator {
						declarations,
						collection,
						where_clause,
					});
				} else {
					// Add destructuring assignments and new where clause
					let origin = EntityRef::new(self.parent.db.upcast(), self.item, patterns[0]);
					if where_clauses.len() == 1 {
						generators.push(Generator::Iterator {
							declarations,
							collection,
							where_clause: Some(where_clauses.pop().unwrap()),
						});
					} else {
						let call = alloc_expression(
							LookupCall {
								function: self.parent.ids.forall.into(),
								arguments: vec![alloc_expression(
									ArrayLiteral(where_clauses),
									self,
									origin,
								)],
							},
							self,
							origin,
						);
						generators.push(Generator::Iterator {
							declarations,
							collection,
							where_clause: Some(call),
						});
					}
					let mut iter = assignments.into_iter();
					let mut assignment = iter.next().unwrap();
					for next in iter {
						generators.push(Generator::Assignment {
							assignment,
							where_clause: None,
						});
						assignment = next;
					}
					generators.push(Generator::Assignment {
						assignment,
						where_clause,
					});
				}
			}
			hir::Generator::Assignment {
				pattern,
				value,
				where_clause,
			} => {
				let def = ExpressionCollector::new(self.parent, self.data, self.item, self.types)
					.collect_expression(*value);
				let assignment = Declaration::from_expression(self.parent.db, false, def);
				let idx = self.parent.model.add_declaration(Item::new(
					assignment,
					EntityRef::new(self.parent.db.upcast(), self.item, *pattern),
				));
				generators.push(Generator::Assignment {
					assignment: idx,
					where_clause: where_clause.map(|w| self.collect_expression(w)),
				});
				let mut asgs = self.collect_destructuring(idx, false, *pattern);
				if !asgs.is_empty() {
					if hir::Pattern::is_refutable(*pattern, self.data) {
						let w = pattern_to_where(
							self,
							idx,
							*pattern,
							EntityRef::new(self.parent.db.upcast(), self.item, *pattern).into(),
						);
						let last = asgs.pop().unwrap();
						generators.extend(asgs.iter().map(|asg| Generator::Assignment {
							assignment: *asg,
							where_clause: None,
						}));
						generators.push(Generator::Assignment {
							assignment: last,
							where_clause: Some(w),
						});
					} else {
						generators.extend(asgs.iter().map(|asg| Generator::Assignment {
							assignment: *asg,
							where_clause: None,
						}));
					}
				}
			}
		}
	}

	fn collect_default_else(&mut self, ty: Ty, origin: Origin) -> Expression {
		let db = self.parent.db;
		match ty.lookup(db.upcast()) {
			TyData::Boolean(_, OptType::Opt)
			| TyData::Integer(_, OptType::Opt)
			| TyData::Float(_, OptType::Opt)
			| TyData::Enum(_, OptType::Opt, _)
			| TyData::Bottom(OptType::Opt)
			| TyData::Array {
				opt: OptType::Opt, ..
			}
			| TyData::Set(_, OptType::Opt, _)
			| TyData::Tuple(OptType::Opt, _)
			| TyData::Record(OptType::Opt, _)
			| TyData::Function(OptType::Opt, _)
			| TyData::TyVar(_, Some(OptType::Opt), _) => alloc_expression(Absent, self, origin),
			TyData::Boolean(_, _) => alloc_expression(BooleanLiteral(true), self, origin),
			TyData::String(_) => alloc_expression(
				StringLiteral::new("", self.parent.db.upcast()),
				self,
				origin,
			),
			TyData::Annotation(_) => {
				alloc_expression(self.parent.ids.empty_annotation, self, origin)
			}
			TyData::Array { .. } => alloc_expression(ArrayLiteral::default(), self, origin),
			TyData::Set(_, _, _) => alloc_expression(SetLiteral::default(), self, origin),
			TyData::Tuple(_, fs) => alloc_expression(
				TupleLiteral(
					fs.iter()
						.map(|f| self.collect_default_else(*f, origin))
						.collect(),
				),
				self,
				origin,
			),
			TyData::Record(_, fs) => alloc_expression(
				RecordLiteral(
					fs.iter()
						.map(|(i, t)| (Identifier(*i), self.collect_default_else(*t, origin)))
						.collect(),
				),
				self,
				origin,
			),
			_ => unreachable!("No default value for this type"),
		}
	}

	// Collect a domain from a user ascribed type
	fn collect_domain(&mut self, t: ArenaIndex<hir::Type>, ty: Ty, is_type_alias: bool) -> Domain {
		let db = self.parent.db;
		let origin = EntityRef::new(db.upcast(), self.item, t);
		match (&self.data[t], ty.lookup(db.upcast())) {
			(hir::Type::Bounded { domain, .. }, _) => {
				if let Some(res) = self.types.name_resolution(*domain) {
					let res_types = db.lookup_item_types(res.item());
					match &res_types[res.pattern()] {
						// Identifier is actually a type, not a domain expression
						PatternTy::TyVar(_) => {
							return Domain::unbounded(self.parent.db, origin, ty);
						}
						PatternTy::TypeAlias { .. } => {
							let model = res.item().model(db.upcast());
							match res.item().local_item_ref(db.upcast()) {
								LocalItemRef::TypeAlias(ta) => {
									let mut c = ExpressionCollector::new(
										self.parent,
										&model[ta].data,
										res.item(),
										&res_types,
									);
									return c.collect_domain(model[ta].aliased_type, ty, true);
								}
								_ => unreachable!(),
							}
						}
						_ => (),
					}
				}
				if is_type_alias {
					// Replace expressions with identifiers pointing to declarations for those expressions
					let er = ExpressionRef::new(self.item, *domain);
					let origin = EntityRef::new(db.upcast(), self.item, *domain);
					Domain::bounded(
						db,
						origin,
						ty.inst(db.upcast()).unwrap(),
						ty.opt(db.upcast()).unwrap(),
						alloc_expression(self.parent.type_alias_expressions[&er], self, origin),
					)
				} else {
					let e = self.collect_expression(*domain);
					Domain::bounded(
						db,
						origin,
						ty.inst(db.upcast()).unwrap(),
						ty.opt(db.upcast()).unwrap(),
						e,
					)
				}
			}
			(
				hir::Type::Array {
					dimensions,
					element,
					..
				},
				TyData::Array {
					opt,
					dim: d,
					element: el,
				},
			) => Domain::array(
				db,
				origin,
				opt,
				self.collect_domain(*dimensions, d, is_type_alias),
				self.collect_domain(*element, el, is_type_alias),
			),
			(
				hir::Type::Set {
					element,
					cardinality,
					..
				},
				TyData::Set(inst, opt, e),
			) => Domain::set_with_card(
				db,
				origin,
				inst,
				opt,
				cardinality.map(|c| self.collect_expression(c)),
				self.collect_domain(*element, e, is_type_alias),
			),
			(hir::Type::Tuple { fields, .. }, TyData::Tuple(opt, fs)) => Domain::tuple(
				db,
				origin,
				opt,
				fields
					.iter()
					.zip(fs.iter())
					.map(|(f, ty)| self.collect_domain(*f, *ty, is_type_alias)),
			),
			(hir::Type::Record { fields, .. }, TyData::Record(opt, fs)) => Domain::record(
				db,
				origin,
				opt,
				fs.iter().map(|(i, ty)| {
					let ident = Identifier(*i);
					(
						ident,
						self.collect_domain(
							fields
								.iter()
								.find_map(|(p, t)| {
									if self.data[*p].identifier().unwrap() == ident {
										Some(*t)
									} else {
										None
									}
								})
								.unwrap(),
							*ty,
							is_type_alias,
						),
					)
				}),
			),
			(hir::Type::New { inst, opt, domain }, _) => {
				// let e = self.collect_expression(*domain);
				// To do:
				// Look up self.item in the enum_idx, recreate constructor expression
				// from the cardinality expression on the item and the corresponding
				// constructor of the class declaration.

				// var new A: x                           -> {<looked up constructor value>}: x
				// var opt new A: x                       -> var opt {<looked up constructor value>}: x
				// var set(d) of new A: x                 -> var set(d) of <looked_up_constructor_value>(1..max(d)): x
				// class B (... var new A: x ...)         ->
				// class B (... var opt new A: x ...)
				// class B (... var set(d) of new A:x ...)

				let (e, new_inst, new_opt) = match self.item.local_item_ref(db.upcast()) {
					LocalItemRef::Declaration(d) => {
						let decl = &self.item.model(db.upcast())[d];
						let item_ty_idx = decl.declared_type;
						let analysis_results = db.class_analysis();
						let (pattern_ref, idx) =
							analysis_results.enum_idx[&PatternRef::new(self.item, decl.pattern)];
						let enum_id = match &self.parent.resolutions[&pattern_ref] {
							LoweredIdentifier::ResolvedIdentifier(
								ResolvedIdentifier::Enumeration(e),
							) => *e,
							_ => unreachable!(),
						};
						let enum_member_id = EnumMemberId::new(enum_id, idx as u32);
						let item_ty = &decl.data[item_ty_idx];
						match item_ty {
							hir::Type::Set { .. } => {
								// let card_expr = self.collect_expression(*c);
								// let origin = card_expr.origin();
								let call = Call {
									function: Callable::EnumConstructor(enum_member_id),
									arguments: vec![],
								};
								let call_expr = alloc_expression(call, self, origin);
								(call_expr, VarType::Par, OptType::NonOpt)
							}
							x => {
								if *opt == OptType::Opt && *inst == VarType::Par {
									let call = Call {
										function: Callable::EnumConstructor(enum_member_id),
										arguments: vec![],
									};
									let call_expr = alloc_expression(call, self, origin);
									(call_expr, VarType::Par, OptType::Opt)	
								} else {
									let ri = ResolvedIdentifier::EnumerationMember(enum_member_id);
									let ri_expr = alloc_expression(ri, self, origin);
									let ri_singleton =
										alloc_expression(SetLiteral(vec![ri_expr]), self, origin);
									let (new_inst, new_opt) = match (inst, opt) {
										(VarType::Par, OptType::NonOpt) => {
											(VarType::Par, OptType::NonOpt)
										}
										(VarType::Par, OptType::Opt) => (VarType::Par, OptType::Opt),
										(VarType::Var, OptType::NonOpt) => {
											(VarType::Par, OptType::NonOpt)
										}
										(VarType::Var, OptType::Opt) => (VarType::Var, OptType::Opt),
									};
									(ri_singleton, new_inst, new_opt)	
								}
							}
						}
					}
					LocalItemRef::Class(c) => todo!(),
					_ => unreachable!(),
				};

				Domain::bounded(db, origin, new_inst, new_opt, e)
			}
			_ => Domain::unbounded(self.parent.db, origin, ty),
		}
	}

	/// Create declarations which perform destructuring according to the given pattern
	fn collect_destructuring(
		&mut self,
		root_decl: DeclarationId,
		top_level: bool,
		pattern: ArenaIndex<hir::Pattern>,
	) -> Vec<DeclarationId> {
		let mut destructuring = Vec::new();
		let mut todo = vec![(0, pattern)];
		while let Some((i, p)) = todo.pop() {
			match &self.data[p] {
				hir::Pattern::Tuple { fields } => {
					for (idx, field) in fields.iter().enumerate() {
						// Destructuring returns the field inside
						destructuring.push(DestructuringEntry::new(
							i,
							Destructuring::TupleAccess(IntegerLiteral(idx as i64 + 1)),
							*field,
						));
						todo.push((destructuring.len(), *field));
					}
				}
				hir::Pattern::Record { fields } => {
					for (ident, field) in fields.iter() {
						// Destructuring returns the field inside
						destructuring.push(DestructuringEntry::new(
							i,
							Destructuring::RecordAccess(*ident),
							*field,
						));
						todo.push((destructuring.len(), *field));
					}
				}
				hir::Pattern::Call {
					function,
					arguments,
				} => {
					let destructuring_pattern = if arguments.len() == 1 {
						// If we have a single arg, destructuring will return the inside directly
						arguments[0]
					} else {
						// Destructuring returns a tuple
						p
					};
					let pat = self.types.pattern_resolution(*function).unwrap();
					let res = &self.parent.resolutions[&pat];
					match res {
						LoweredIdentifier::Callable(Callable::Annotation(ann)) => {
							destructuring.push(DestructuringEntry::new(
								i,
								Destructuring::Annotation(*ann),
								destructuring_pattern,
							));
						}
						LoweredIdentifier::Callable(Callable::EnumConstructor(member)) => {
							destructuring.push(DestructuringEntry::new(
								i,
								Destructuring::Enumeration(*member),
								destructuring_pattern,
							));
						}
						_ => unreachable!(),
					};
					let j = destructuring.len();
					if arguments.len() == 1 {
						todo.push((j, arguments[0]));
					} else {
						for (idx, field) in arguments.iter().enumerate() {
							// Destructuring the tuple returns the field inside
							destructuring.push(DestructuringEntry::new(
								j,
								Destructuring::TupleAccess(IntegerLiteral(idx as i64 + 1)),
								*field,
							));
							todo.push((destructuring.len(), *field));
						}
					}
				}
				hir::Pattern::Identifier(name) => {
					if matches!(
						&self.types[p],
						PatternTy::Variable(_) | PatternTy::Argument(_)
					) {
						if i > 0 {
							destructuring[i - 1].name = Some(*name);
							// Mark used destructurings as to be created
							let mut c = i;
							loop {
								if c == 0 {
									break;
								}
								let item = &mut destructuring[c - 1];
								if item.create {
									break;
								}
								item.create = true;
								c = item.parent;
							}
						} else {
							self.parent.model[root_decl].set_name(*name);
							self.parent.resolutions.insert(
								PatternRef::new(self.item, pattern),
								LoweredIdentifier::ResolvedIdentifier(root_decl.into()),
							);
						}
					}
				}
				_ => (),
			}
		}
		let mut decls = Vec::new();
		let mut decl_map = FxHashMap::default();
		for (idx, item) in destructuring
			.into_iter()
			.enumerate()
			.filter(|(_, item)| item.create)
		{
			let origin = EntityRef::new(self.parent.db.upcast(), self.item, item.pattern);
			let decl = self.introduce_declaration(top_level, origin, |collector| {
				let ident = alloc_expression(
					if item.parent == 0 {
						root_decl
					} else {
						decl_map[&item.parent]
					},
					collector,
					origin,
				);
				match item.kind {
					Destructuring::Annotation(a) => alloc_expression(
						Call {
							function: Callable::AnnotationDestructure(a),
							arguments: vec![ident],
						},
						collector,
						origin,
					),
					Destructuring::Enumeration(e) => alloc_expression(
						Call {
							function: Callable::EnumDestructor(e),
							arguments: vec![ident],
						},
						collector,
						origin,
					),
					Destructuring::RecordAccess(f) => alloc_expression(
						RecordAccess {
							record: Box::new(ident),
							field: f,
						},
						collector,
						origin,
					),
					Destructuring::TupleAccess(f) => alloc_expression(
						TupleAccess {
							tuple: Box::new(ident),
							field: f,
						},
						collector,
						origin,
					),
				}
			});
			if let Some(name) = item.name {
				self.parent.model[decl].set_name(name);
				self.parent.resolutions.insert(
					PatternRef::new(self.item, item.pattern),
					LoweredIdentifier::ResolvedIdentifier(decl.into()),
				);
			}
			decl_map.insert(idx + 1, decl);
			decls.push(decl);
		}
		decls
	}

	/// Lower an HIR pattern into a THIR pattern
	fn collect_pattern(&mut self, pattern: ArenaIndex<hir::Pattern>) -> Pattern {
		let db = self.parent.db;
		let origin = EntityRef::new(db.upcast(), self.item, pattern);
		let ty = match &self.types[pattern] {
			PatternTy::Destructuring(ty) => *ty,
			PatternTy::Variable(ty) | PatternTy::Argument(ty) => {
				return Pattern::anonymous(*ty, origin);
			}
			_ => unreachable!(),
		};
		match &self.data[pattern] {
			hir::Pattern::Absent => {
				Pattern::expression(alloc_expression(Absent, self, origin), origin)
			}
			hir::Pattern::Anonymous => Pattern::anonymous(ty, origin),
			hir::Pattern::Boolean(b) => {
				Pattern::expression(alloc_expression(*b, self, origin), origin)
			}
			hir::Pattern::Call {
				function,
				arguments,
			} => {
				let args = arguments
					.iter()
					.map(|a| self.collect_pattern(*a))
					.collect::<Vec<_>>();
				let pat = self.types.pattern_resolution(*function).unwrap();
				let res = &self.parent.resolutions[&pat];
				match res {
					LoweredIdentifier::Callable(Callable::Annotation(ann)) => {
						Pattern::annotation_constructor(db, &self.parent.model, origin, *ann, args)
					}
					LoweredIdentifier::Callable(Callable::EnumConstructor(member)) => {
						Pattern::enum_constructor(db, &self.parent.model, origin, *member, args)
					}
					_ => unreachable!(),
				}
			}
			hir::Pattern::Float { negated, value } => {
				let v = alloc_expression(*value, self, origin);
				Pattern::expression(
					if *negated {
						alloc_expression(
							LookupCall {
								function: self.parent.ids.minus.into(),
								arguments: vec![v],
							},
							self,
							origin,
						)
					} else {
						v
					},
					origin,
				)
			}
			hir::Pattern::Identifier(_) => {
				let pat = self.types.pattern_resolution(pattern).unwrap();
				let res = &self.parent.resolutions[&pat];
				match res {
					LoweredIdentifier::ResolvedIdentifier(ResolvedIdentifier::Annotation(a)) => {
						Pattern::expression(alloc_expression(*a, self, origin), origin)
					}
					LoweredIdentifier::ResolvedIdentifier(
						ResolvedIdentifier::EnumerationMember(m),
					) => Pattern::expression(alloc_expression(*m, self, origin), origin),
					_ => unreachable!(),
				}
			}
			hir::Pattern::Infinity { negated } => {
				let v = alloc_expression(Infinity, self, origin);
				Pattern::expression(
					if *negated {
						alloc_expression(
							LookupCall {
								function: self.parent.ids.minus.into(),
								arguments: vec![v],
							},
							self,
							origin,
						)
					} else {
						v
					},
					origin,
				)
			}
			hir::Pattern::Integer { negated, value } => {
				let v = alloc_expression(*value, self, origin);
				Pattern::expression(
					if *negated {
						alloc_expression(
							LookupCall {
								function: self.parent.ids.minus.into(),
								arguments: vec![v],
							},
							self,
							origin,
						)
					} else {
						v
					},
					origin,
				)
			}
			hir::Pattern::Missing => unreachable!(),
			hir::Pattern::Record { fields } => {
				let fields = fields
					.iter()
					.map(|(i, p)| (*i, self.collect_pattern(*p)))
					.collect::<Vec<_>>();
				Pattern::record(db, &self.parent.model, origin, fields)
			}
			hir::Pattern::String(s) => {
				Pattern::expression(alloc_expression(s.clone(), self, origin), origin)
			}
			hir::Pattern::Tuple { fields } => {
				let fields = fields
					.iter()
					.map(|f| self.collect_pattern(*f))
					.collect::<Vec<_>>();
				Pattern::tuple(db, &self.parent.model, origin, fields)
			}
		}
	}
}

fn alloc_expression(
	data: impl ExpressionBuilder,
	collector: &ExpressionCollector<'_, '_>,
	origin: impl Into<Origin>,
) -> Expression {
	Expression::new(collector.parent.db, &collector.parent.model, origin, data)
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct DestructuringEntry {
	parent: usize, // 0 means no parent, otherwise = index of parent + 1
	kind: Destructuring,
	pattern: ArenaIndex<hir::Pattern>,
	name: Option<Identifier>,
	create: bool,
}

impl DestructuringEntry {
	fn new(parent: usize, kind: Destructuring, pattern: ArenaIndex<hir::Pattern>) -> Self {
		Self {
			parent,
			kind,
			pattern,
			name: None,
			create: false,
		}
	}
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Destructuring {
	TupleAccess(IntegerLiteral),
	RecordAccess(Identifier),
	Enumeration(EnumMemberId),
	Annotation(AnnotationId),
}

/// Lower a model to THIR
pub fn lower_model(db: &dyn Thir) -> Arc<Intermediate<Model>> {
	log::info!("Lowering model to THIR");
	let items = db.run_hir_phase().unwrap_or_else(|e| {
		panic!(
			"Errors present, cannot lower model.\n{}",
			e.iter()
				.map(|e| format!("{:#?}", e))
				.collect::<Vec<_>>()
				.join("\n")
		)
	});
	let ids = db.identifier_registry();
	let counts = db.entity_counts();
	let mut collector = ItemCollector::new(db, &ids, &counts);
	for item in items.iter() {
		collector.collect_item(*item);
	}
	collector.collect_deferred();
	Arc::new(Intermediate::new(collector.finish()))
}
