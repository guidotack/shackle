//! Converting objects of a subclass to objects of a superclass
//!
//! Every class has its own enum of potential objects, so an object of a
//! subclass `B` used where its superclass `A` is expected must become the
//! identity the same object has in `A`'s enum. This is the class counterpart of
//! the automatic coercions `top_down_type` makes explicit, but it has to happen
//! during lowering: afterwards `B_potential` and `A_potential` are unrelated
//! enums, and the MiniZinc target prints the model before any transform runs.
//!
//! Each occurrence contributes one constructor to the enum of its class and of
//! every superclass, over the same domain, so the conversion is
//! `B_occ_k(i) ↦ A_occ_k'(i)`: a par array `B_to_A`, indexed by `B`'s enum and
//! defined in `finish` once every enum is complete. It is applied wherever the
//! type checker expects a superclass, lifted through opt, sets, arrays, tuples
//! and records.

use shackle_hir::{IntegerLiteral, class_analysis::class_pattern_for, ids::PatternRef};
use shackle_ty::{ClassRef, OptType, Ty, TyData, VarType};

use crate::{
	lower::{
		ItemCollector,
		expression::{ExpressionCollector, alloc_expression},
	},
	*,
};

/// Whether converting a value of type `from` to type `to` changes the class of
/// an object in it
fn changes_class<'db>(db: &'db dyn Db, from: Ty<'db>, to: Ty<'db>) -> bool {
	match (from.lookup(db), to.lookup(db)) {
		(TyData::Class(_, _, a), TyData::Class(_, _, b)) => a != b,
		(TyData::Set(_, _, a), TyData::Set(_, _, b)) => changes_class(db, *a, *b),
		(TyData::Array { element: a, .. }, TyData::Array { element: b, .. }) => {
			changes_class(db, *a, *b)
		}
		(TyData::Tuple(_, a), TyData::Tuple(_, b)) => {
			a.len() == b.len()
				&& a.iter()
					.zip(b.iter())
					.any(|(a, b)| changes_class(db, *a, *b))
		}
		(TyData::Record(_, a), TyData::Record(_, b)) => a.iter().any(|(name, a)| {
			b.iter()
				.find(|(n, _)| n == name)
				.is_some_and(|(_, b)| changes_class(db, *a, *b))
		}),
		_ => false,
	}
}

impl<'db, 'a, 'b, 'c> ExpressionCollector<'db, 'a, 'b, 'c> {
	/// Collect `idx` where the type checker expects a value of type
	/// `expected`, converting the objects of a subclass in it to objects of the
	/// expected superclass
	pub(in crate::lower) fn collect_expression_upcast(
		&mut self,
		idx: shackle_hir::ExpressionId<'db>,
		expected: Ty<'db>,
	) -> Expression<'db> {
		let expr = self.collect_expression(idx);
		self.upcast(expr, self.types[idx], expected)
	}

	/// Convert `expr`, of type `from`, to type `to`: every object of a subclass
	/// in it becomes the same object of the superclass `to` names
	pub(in crate::lower) fn upcast(
		&mut self,
		expr: Expression<'db>,
		from: Ty<'db>,
		to: Ty<'db>,
	) -> Expression<'db> {
		let db = self.parent.db;
		if !changes_class(db, from, to) {
			return expr;
		}
		let expr = self.relabel_class_operand(expr);
		if from.opt(db) == Some(OptType::Opt) {
			// `if occurs(x) then <x converted> else <> endif`
			let (from, to) = (from.make_occurs(db), to.make_occurs(db));
			return self.bind(expr, |this, x| {
				let origin = x.origin();
				let occurs = alloc_expression(
					LookupCall {
						function: this.parent.ids.functions.occurs.into(),
						arguments: vec![x.clone()],
					},
					this,
					origin,
				);
				let value = alloc_expression(
					LookupCall {
						function: this.parent.ids.functions.deopt.into(),
						arguments: vec![x],
					},
					this,
					origin,
				);
				let converted = this.upcast(value, from, to);
				let absent = alloc_expression(Absent, this, origin);
				alloc_expression(
					IfThenElse {
						branches: vec![Branch {
							condition: occurs,
							result: converted,
						}],
						else_result: Box::new(absent),
					},
					this,
					origin,
				)
			});
		}
		let origin = expr.origin();
		match (from.lookup(db), to.lookup(db)) {
			(TyData::Class(_, _, sub), TyData::Class(_, _, sup)) => {
				// `B_to_A[x]`
				let map = self.parent.upcast_map(*sub, *sup);
				let map = alloc_expression(map, self, origin);
				self.introduce_array_access(map, expr, origin)
			}
			(TyData::Set(_, _, from_element), TyData::Set(_, _, to_element)) => {
				// `mzn_injective_image(s, B_to_A)`
				let (Some(sub), Some(sup)) =
					(from_element.class_type(db), to_element.class_type(db))
				else {
					unreachable!("a set of objects holds objects")
				};
				let map = self.parent.upcast_map(sub, sup);
				let map = alloc_expression(map, self, origin);
				alloc_expression(
					LookupCall {
						function: self.parent.ids.functions.mzn_injective_image.into(),
						arguments: vec![expr, map],
					},
					self,
					origin,
				)
			}
			(TyData::Array { element: from, .. }, TyData::Array { element: to, .. }) => {
				// `arrayXd(x, [<e converted> | e in x])`
				let (from, to) = (*from, *to);
				self.bind(expr, |this, x| {
					let element_ty = x.ty().elem_ty(db).expect("an array has elements");
					let element =
						Declaration::new(false, Domain::unbounded(db, origin, element_ty));
					let element = this
						.parent
						.model
						.add_declaration(DeclarationItem::new(element, origin));
					let template = alloc_expression(element, this, origin);
					let template = this.upcast(template, from, to);
					let comprehension = alloc_expression(
						ArrayComprehension {
							indices: None,
							template: Box::new(template),
							generators: vec![Generator::Iterator {
								declarations: vec![element],
								collection: x.clone(),
								where_clause: None,
							}],
						},
						this,
						origin,
					);
					alloc_expression(
						LookupCall {
							function: this.parent.ids.functions.array_xd.into(),
							arguments: vec![x, comprehension],
						},
						this,
						origin,
					)
				})
			}
			(TyData::Tuple(_, from), TyData::Tuple(_, to)) => {
				let fields = from
					.iter()
					.copied()
					.zip(to.iter().copied())
					.collect::<Vec<_>>();
				self.bind(expr, |this, x| {
					let fields = fields
						.into_iter()
						.enumerate()
						.map(|(i, (from, to))| {
							let field = alloc_expression(
								TupleAccess {
									tuple: Box::new(x.clone()),
									field: IntegerLiteral(i as i64 + 1),
								},
								this,
								origin,
							);
							this.upcast(field, from, to)
						})
						.collect();
					alloc_expression(TupleLiteral(fields), this, origin)
				})
			}
			(TyData::Record(_, from), TyData::Record(_, to)) => {
				let fields = from
					.iter()
					.map(|(name, from)| {
						let to = to
							.iter()
							.find(|(n, _)| n == name)
							.map_or(*from, |(_, to)| *to);
						(*name, *from, to)
					})
					.collect::<Vec<_>>();
				self.bind(expr, |this, x| {
					let fields = fields
						.into_iter()
						.map(|(name, from, to)| {
							let name: Identifier<'db> = name.into();
							let field = alloc_expression(
								RecordAccess {
									record: Box::new(x.clone()),
									field: name,
								},
								this,
								origin,
							);
							(name, this.upcast(field, from, to))
						})
						.collect();
					alloc_expression(RecordLiteral(fields), this, origin)
				})
			}
			_ => expr,
		}
	}

	/// `let { x = expr } in body(x)`, so that `body` can use the value of
	/// `expr` more than once
	fn bind(
		&mut self,
		expr: Expression<'db>,
		body: impl FnOnce(&mut Self, Expression<'db>) -> Expression<'db>,
	) -> Expression<'db> {
		let origin = expr.origin();
		let decl = self.introduce_declaration(false, origin, |_| expr);
		let x = alloc_expression(decl, self, origin);
		let in_expression = body(self, x);
		alloc_expression(
			Let {
				items: vec![LetItem::Declaration(decl)],
				in_expression: Box::new(in_expression),
			},
			self,
			origin,
		)
	}
}

impl<'db> ItemCollector<'db> {
	/// The declaration of `array [<sub>_potential] of <sup>_potential:
	/// <sub>_to_<sup>`, which maps each object of class `sub` to the same object
	/// of its superclass `sup`. It is defined in `finish` by
	/// `define_upcast_maps`, once every class enum is complete.
	pub(in crate::lower) fn upcast_map(
		&mut self,
		sub: ClassRef<'db>,
		sup: ClassRef<'db>,
	) -> DeclarationId<'db> {
		let db = self.db;
		let sub = class_pattern_for(db, sub).expect("class item for class type");
		let sup = class_pattern_for(db, sup).expect("class item for class type");
		if let Some((_, _, map)) = self
			.objects
			.upcast_maps
			.iter()
			.find(|(s, t, _)| *s == sub && *t == sup)
		{
			return *map;
		}
		self.ensure_class_predeclared(sub);
		self.ensure_class_predeclared(sup);
		let item = sub.item(db);
		let sub_enum = self.objects.class_map[&sub].class_enum;
		let sup_enum = self.objects.class_map[&sup].class_enum;
		let dim = Domain::bounded(
			db,
			item,
			VarType::Par,
			OptType::NonOpt,
			Expression::new(db, &self.model, item, sub_enum),
		);
		let element =
			Domain::unbounded(db, item, Ty::par_enum(db, self.model[sup_enum].enum_type()));
		let mut decl =
			Declaration::new(true, Domain::array(db, item, OptType::NonOpt, dim, element));
		decl.set_name(Identifier::new(
			db,
			format!("{}_to_{}", class_name(db, sub), class_name(db, sup)),
		));
		decl.annotations_mut().push(Expression::new(
			db,
			&self.model,
			item,
			self.ids.annotations.no_output,
		));
		let map = self.model.add_declaration(DeclarationItem::new(decl, item));
		self.objects.upcast_maps.push((sub, sup, map));
		map
	}

	/// Define the maps `upcast_map` declared: each constructor `<sub>_occ_k`
	/// of the subclass's enum maps to the constructor `<sup>_occ_k'` the same
	/// occurrence contributes to the superclass, argument for argument
	pub(in crate::lower) fn define_upcast_maps(&mut self) {
		let db = self.db;
		for (sub, sup, map) in self.objects.upcast_maps.clone() {
			let item = sub.item(db);
			let sub_enum = self.objects.class_map[&sub].class_enum;
			let sup_enum = self.objects.class_map[&sup].class_enum;
			let constructor_count = self.model[sub_enum]
				.definition()
				.map_or(0, |constructors| constructors.len());
			let mut blocks: Option<Expression<'db>> = None;
			for constructor_index in 0..constructor_count {
				let occurrence = self
					.objects
					.plan
					.contributions_in_occurrence_order()
					.flatten()
					.find(|c| c.target_class == sub && c.constructor_index == constructor_index)
					.expect("every constructor of a class enum comes from an occurrence")
					.occurrence;
				let target = self.objects.plan.contributions_by_occurrence[&occurrence]
					.iter()
					.find(|c| c.target_class == sup)
					.expect("an occurrence contributes to every superclass of its class")
					.constructor_index;
				let domain = self
					.class_enum_constructor_domain(sub_enum, constructor_index)
					.expect("object constructor has a bounded parameter domain");
				// `[<sup>_occ_k'(i) | i in <domain>]`
				let i = Declaration::new(false, Domain::unbounded(db, item, Ty::par_int(db)));
				let i = self.model.add_declaration(DeclarationItem::new(i, item));
				let template = Expression::new(
					db,
					&self.model,
					item,
					Call {
						function: Callable::EnumConstructor(EnumMemberId::new(
							sup_enum,
							target as u32,
						)),
						arguments: vec![Expression::new(db, &self.model, item, i)],
					},
				);
				let block = Expression::new(
					db,
					&self.model,
					item,
					ArrayComprehension {
						indices: None,
						template: Box::new(template),
						generators: vec![Generator::Iterator {
							declarations: vec![i],
							collection: domain,
							where_clause: None,
						}],
					},
				);
				blocks = Some(match blocks {
					None => block,
					Some(blocks) => Expression::new(
						db,
						&self.model,
						item,
						LookupCall {
							function: self.ids.functions.plus_plus.into(),
							arguments: vec![blocks, block],
						},
					),
				});
			}
			let definition = match blocks {
				Some(blocks) => {
					let potential = Expression::new(db, &self.model, item, sub_enum);
					Expression::new(
						db,
						&self.model,
						item,
						LookupCall {
							function: self.ids.functions.array1d.into(),
							arguments: vec![potential, blocks],
						},
					)
				}
				None => Expression::new_unchecked(self.model[map].ty(), ArrayLiteral(vec![]), item),
			};
			self.model[map].set_definition(definition);
		}
	}
}

/// The name of a class, for the names of generated declarations
fn class_name<'db>(db: &'db dyn Db, class: PatternRef<'db>) -> &'db str {
	class.identifier(db).expect("a class has a name").lookup(db)
}
