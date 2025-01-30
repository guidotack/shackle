/// Types of signatures - the type required when referring to an item
///
/// E.g.
/// - Function parameter/return type
/// - Variable declaration LHS types
use rustc_hash::FxHashMap;

use super::{EnumConstructorEntry, PatternTy, TypeCompletionMode, TypeContext, Typer};
use crate::{
	diagnostics::{SyntaxError, TypeInferenceFailure, TypeMismatch},
	hir::{
		db::Hir,
		ids::{EntityRef, ExpressionRef, ItemRef, LocalItemRef, NodeRef, PatternRef, TypeRef},
		ClassItem, Constructor, ConstructorParameter, EnumConstructor, Goal, Identifier, ItemData,
		Pattern, Type,
	},
	ty::{
		ClassRef, EnumRef, FunctionEntry, FunctionType, OverloadedFunction,
		PolymorphicFunctionType, Ty, TyData, TyVar, TyVarRef,
	},
	Error,
};

/// Collected types for an item signature
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SignatureTypes {
	/// Types of declarations
	pub patterns: FxHashMap<PatternRef, PatternTy>,
	/// Types of expressions
	pub expressions: FxHashMap<ExpressionRef, Ty>,
	/// Identifier resolution
	pub identifier_resolution: FxHashMap<ExpressionRef, PatternRef>,
	/// Pattern resolution
	pub pattern_resolution: FxHashMap<PatternRef, PatternRef>,
	/// Types of TypeRefs
	pub types: FxHashMap<TypeRef, Ty>,
}

/// Context for typing an item signature
pub struct SignatureTypeContext {
	starting_item: ItemRef,
	data: SignatureTypes,
	diagnostics: Vec<Error>,
}

impl SignatureTypeContext {
	/// Create a new signature type context
	pub fn new(item: ItemRef) -> Self {
		Self {
			starting_item: item,
			data: SignatureTypes {
				patterns: FxHashMap::default(),
				expressions: FxHashMap::default(),
				identifier_resolution: FxHashMap::default(),
				pattern_resolution: FxHashMap::default(),
				types: FxHashMap::default(),
			},
			diagnostics: Vec::new(),
		}
	}

	/// Compute the signature of the given item
	pub fn type_item(&mut self, db: &dyn Hir, item: ItemRef) {
		let model = &*item.model(db);
		let it = item.local_item_ref(db);
		let data = it.data(model);
		match it {
			LocalItemRef::Annotation(a) => {
				let it = &model[a];
				match &it.constructor {
					Constructor::Atom { pattern } => {
						self.add_declaration(
							PatternRef::new(item, *pattern),
							PatternTy::AnnotationAtom,
						);
					}
					Constructor::Function {
						constructor,
						destructor,
						parameters,
					} => {
						let params = parameters
							.iter()
							.map(|p| {
								let mut had_error = false;
								for t in Type::any_types(p.declared_type, &it.data) {
									let (src, span) =
										NodeRef::from(EntityRef::new(db, item, t)).source_span(db);
									self.add_diagnostic(
										item,
										TypeInferenceFailure {
											src,
											span,
											msg: "Incomplete parameter types are not allowed"
												.to_owned(),
										},
									);
									had_error = true;
								}
								let ty = if had_error {
									db.type_registry().error
								} else {
									Typer::new(db, self, item, data)
										.complete_type(
											p.declared_type,
											None,
											TypeCompletionMode::AnnotationParameter,
										)
										.ty
								};
								if let Some(pat) = p.pattern {
									self.add_declaration(
										PatternRef::new(item, pat),
										PatternTy::Argument(ty),
									);
								}
								ty
							})
							.collect::<Box<_>>();
						let ann = db.type_registry().ann;
						if !params.is_empty() {
							let dtor = FunctionEntry {
								has_body: false,
								overload: OverloadedFunction::Function(FunctionType {
									return_type: if params.len() == 1 {
										params[0]
									} else {
										Ty::tuple(db.upcast(), params.iter().copied())
									},
									params: Box::new([ann]),
								}),
							};
							self.add_declaration(
								PatternRef::new(item, *destructor),
								PatternTy::AnnotationDestructure(Box::new(dtor)),
							);
						}
						let ctor = FunctionEntry {
							has_body: false,
							overload: OverloadedFunction::Function(FunctionType {
								return_type: ann,
								params,
							}),
						};
						self.add_declaration(
							PatternRef::new(item, *constructor),
							PatternTy::AnnotationConstructor(Box::new(ctor)),
						);
					}
				}
			}
			LocalItemRef::Function(f) => {
				let it = &model[f];
				// Set as computing so if there's a call to a function with this name we can break the cycle
				// (since if the call is actually not referring to this overload, it should work)
				self.add_declaration(PatternRef::new(item, it.pattern), PatternTy::Computing);
				let ty_params = it
					.type_inst_vars
					.iter()
					.map(|tv| {
						let ty_var = TyVarRef::new(db, PatternRef::new(item, tv.name));
						let type_var = TyVar {
							ty_var,
							varifiable: tv.is_varifiable,
							enumerable: tv.is_enum,
							indexable: tv.is_indexable,
						};
						self.add_declaration(
							PatternRef::new(item, tv.name),
							PatternTy::TyVar(type_var),
						);
						ty_var
					})
					.collect::<Box<[_]>>();
				let params = it
					.parameters
					.iter()
					.enumerate()
					.map(|(i, p)| {
						let mut had_error = false;
						let annotated_expression = p
							.annotations
							.iter()
							.find(|ann| match &it.data[**ann] {
								crate::hir::Expression::Identifier(i) => {
									*i == db.identifier_registry().annotated_expression
								}
								_ => false,
							})
							.copied();
						if i > 0 {
							if let Some(ann) = annotated_expression {
								let (src, span) =
									NodeRef::from(EntityRef::new(db, item, ann)).source_span(db);
								self.add_diagnostic(
									item,
									SyntaxError {
										src,
										span,
										msg: "'annotated_expression' only allowed on first function parameter.".to_owned(),
										other: Vec::new(),
									},
								);
							}
						}
						for t in Type::any_types(p.declared_type, &it.data) {
							let (src, span) =
								NodeRef::from(EntityRef::new(db, item, t)).source_span(db);
							self.add_diagnostic(
								item,
								TypeInferenceFailure {
									src,
									span,
									msg: "Incomplete parameter types are not allowed".to_owned(),
								},
							);
							had_error = true;
						}
						let mut typer = Typer::new(db, self, item, data);
						let ty = if had_error {
							db.type_registry().error
						} else {
							typer
								.complete_type(p.declared_type, None, TypeCompletionMode::Default)
								.ty
						};
						if let Some(pat) = p.pattern {
							typer.collect_pattern(None, false, pat, ty, true);
						}
						ty
					})
					.collect();
				let pattern = PatternRef::new(item, it.pattern);
				if ty_params.is_empty() {
					let f = FunctionType {
						return_type: db.type_registry().error,
						params,
					};
					self.add_declaration(
						pattern,
						PatternTy::Function(Box::new(FunctionEntry {
							has_body: it.body.is_some(),
							overload: OverloadedFunction::Function(f),
						})),
					);
				} else {
					let p = PolymorphicFunctionType {
						return_type: db.type_registry().error,
						ty_params,
						params,
					};
					self.add_declaration(
						pattern,
						PatternTy::Function(Box::new(FunctionEntry {
							has_body: it.body.is_some(),
							overload: OverloadedFunction::PolymorphicFunction(p),
						})),
					);
				}

				let mut had_error = false;
				for t in Type::any_types(it.return_type, &it.data)
					.chain(Type::anonymous_ty_vars(it.return_type, &it.data))
				{
					let (src, span) = NodeRef::from(EntityRef::new(db, item, t)).source_span(db);
					self.add_diagnostic(
						item,
						TypeInferenceFailure {
							src,
							span,
							msg: "Incomplete return type not allowed".to_owned(),
						},
					);
					had_error = true;
				}
				let return_type = if had_error {
					db.type_registry().error
				} else {
					Typer::new(db, self, item, data)
						.complete_type(it.return_type, None, TypeCompletionMode::Default)
						.ty
				};

				let d = self.data.patterns.get_mut(&pattern).unwrap();
				match d {
					PatternTy::Function(function) => match function.as_mut() {
						FunctionEntry {
							overload: OverloadedFunction::Function(f),
							..
						} => {
							f.return_type = return_type;
						}
						FunctionEntry {
							overload: OverloadedFunction::PolymorphicFunction(p),
							..
						} => {
							p.return_type = return_type;
						}
					},
					_ => unreachable!(),
				}
			}
			LocalItemRef::Declaration(d) => {
				let it = &model[d];
				let ids = db.identifier_registry();
				let output_only = it
					.annotations
					.iter()
					.find(|ann| match &it.data[**ann] {
						crate::hir::Expression::Identifier(i) => *i == ids.output_only,
						_ => false,
					})
					.copied();
				for p in Pattern::identifiers(it.pattern, data) {
					self.add_declaration(PatternRef::new(item, p), PatternTy::Computing);
				}
				let mut typer = Typer::new(db, self, item, data);
				let ty = if data[it.declared_type].is_complete(data) {
					// Use LHS type only
					let expected = typer
						.complete_type(it.declared_type, None, TypeCompletionMode::Default)
						.ty;
					typer.collect_pattern(None, false, it.pattern, expected, false)
				} else if output_only.is_some() {
					typer.collect_output_declaration(it)
				} else {
					typer.collect_declaration(it)
				};

				if it.definition.is_none()
					&& (ty.contains_var(db.upcast()) && ty.contains_par(db.upcast())
						|| ty.contains_function(db.upcast()))
				{
					let (src, span) = NodeRef::from(item).source_span(db);
					self.add_diagnostic(
						item,
						SyntaxError {
							src,
							span,
							msg: "declaration must have a right-hand side.".to_owned(),
							other: Vec::new(),
						},
					);
				}

				if let Some(ann) = output_only {
					if it.definition.is_none() {
						let (src, span) =
							NodeRef::from(EntityRef::new(db, item, ann)).source_span(db);
						self.add_diagnostic(
							item,
							SyntaxError {
								src,
								span,
								msg: "'output_only' declarations must have a right-hand side."
									.to_owned(),
								other: Vec::new(),
							},
						);
					}
					if !ty.known_par(db.upcast()) {
						let (src, span) =
							NodeRef::from(EntityRef::new(db, item, ann)).source_span(db);
						self.add_diagnostic(
							item,
							TypeMismatch {
								src,
								span,
								msg: "'output_only' declarations must be par.".to_owned(),
							},
						);
					}
				}
			}
			LocalItemRef::Enumeration(e) => {
				let it = &model[e];
				let ty = Ty::par_enum(
					db.upcast(),
					EnumRef::new(db, PatternRef::new(item, it.pattern)),
				);
				self.add_declaration(
					PatternRef::new(item, it.pattern),
					PatternTy::Enum(Ty::par_set(db.upcast(), ty).unwrap()),
				);
				if let Some(cases) = &it.definition {
					self.add_enum_cases(db, item, data, ty, cases);
				}
			}
			LocalItemRef::EnumAssignment(e) => {
				let it = &model[e];
				let set_ty = Typer::new(db, self, item, data).collect_expression(it.assignee);
				let ty = match set_ty.lookup(db.upcast()) {
					TyData::Set(_, _, e) => e,
					_ => unreachable!(),
				};
				self.add_enum_cases(db, item, data, ty, &it.definition);
			}
			LocalItemRef::Solve(s) => {
				let it = &model[s];
				match &it.goal {
					Goal::Maximize { pattern, objective }
					| Goal::Minimize { pattern, objective } => {
						self.add_declaration(PatternRef::new(item, *pattern), PatternTy::Computing);
						let actual =
							Typer::new(db, self, item, data).collect_expression(*objective);
						if !actual.is_subtype_of(db.upcast(), db.type_registry().var_float) {
							let (src, span) =
								NodeRef::from(EntityRef::new(db, item, *objective)).source_span(db);
							self.add_diagnostic(
								item,
								TypeMismatch {
									src,
									span,
									msg: format!(
										"Objective must be numeric, but got '{}'",
										actual.pretty_print(db.upcast())
									),
								},
							);
						}
						self.add_declaration(
							PatternRef::new(item, *pattern),
							PatternTy::Variable(actual),
						);
					}
					_ => (),
				}
			}
			LocalItemRef::TypeAlias(t) => {
				let it = &model[t];
				let pat = PatternRef::new(item, it.name);
				self.add_declaration(pat, PatternTy::Computing);
				let result = Typer::new(db, self, item, data).complete_type(
					it.aliased_type,
					None,
					TypeCompletionMode::Default,
				);
				self.add_declaration(
					pat,
					PatternTy::TypeAlias {
						ty: result.ty,
						has_bounded: result.has_bounded,
						has_unbounded: result.has_unbounded,
					},
				);
			}
			LocalItemRef::Class(c) => {
				let itemref = item;
				let it = &model[c];
				let pat = PatternRef::new(item, it.pattern);
				let tys = db.type_registry();
				// Create empty class decl type
				let mut class_decl_type = ClassRef::new(db, pat);

				self.add_declaration(pat, PatternTy::Computing);

				let mut record_ty_fields = Vec::new();

				if let Some(base) = it.extends {
					let mut typer = Typer::new(db, self, itemref, data);
					let base_type = typer.collect_expression(base);
					if let Some(class_type) = base_type.class_type(db.upcast()) {
						class_decl_type.superclass = Some(base_type);
						match self.type_pattern(db, class_type.pattern(db.upcast())) {
							PatternTy::ClassDecl {
								input_record_ty, ..
							} => {
								if let Some(fields) = input_record_ty.record_fields(db.upcast()) {
									record_ty_fields.extend(
										fields.into_iter().map(|(is, ty)| (Identifier(is), ty)),
									);
								}
							}
							_ => unreachable!(),
						}
					} else {
						let (src, span) =
							NodeRef::from(EntityRef::new(db, item, base)).source_span(db);
						self.add_diagnostic(
							item,
							TypeMismatch {
								src,
								span,
								msg: format!(
									"Expected class, but got '{}'",
									base_type.pretty_print(db.upcast())
								),
							},
						);
					}
				}

				let error_ty = db.type_registry().error;

				self.add_declaration(
					pat,
					PatternTy::ClassDecl {
						defining_set_ty: Ty::par_set(
							db.upcast(),
							Ty::class(db.upcast(), class_decl_type.clone()),
						)
						.unwrap(),
						input_record_ty: error_ty,
					},
				);

				for item in it.items.iter() {
					match item {
						ClassItem::Constraint(c) => {
							let mut typer = Typer::new(db, self, itemref, data);
							for ann in c.annotations.iter() {
								typer.typecheck_expression(*ann, tys.ann);
							}
							typer.typecheck_expression(c.expression, tys.var_bool);
						}
						ClassItem::Declaration(d) => {
							let mut typer = Typer::new(db, self, itemref, data);
							let field_name =
								PatternRef::new(itemref, d.pattern).identifier(db).unwrap();
							let ty = typer.collect_declaration(d);
							if let Some(record_ty) =
								typer.class_type_to_input_record_type(d.declared_type)
							{
								record_ty_fields.push((field_name, record_ty));
							}
							class_decl_type.attributes.push((field_name, ty));
							self.add_declaration(
								pat,
								PatternTy::ClassDecl {
									defining_set_ty: Ty::par_set(
										db.upcast(),
										Ty::class(db.upcast(), class_decl_type.clone()),
									)
									.unwrap(),
									input_record_ty: error_ty,
								},
							);
						}
					}
				}

				self.add_declaration(
					pat,
					PatternTy::ClassDecl {
						defining_set_ty: Ty::par_set(
							db.upcast(),
							Ty::class(db.upcast(), class_decl_type.clone()),
						)
						.unwrap(),
						input_record_ty: Ty::record(db.upcast(), record_ty_fields),
					},
				);
			}
			_ => unreachable!("Item {:?} does not have signature", it),
		}
	}

	fn add_enum_cases(
		&mut self,
		db: &dyn Hir,
		item: ItemRef,
		data: &ItemData,
		ty: Ty,
		cases: &[EnumConstructor],
	) {
		let get_param_types = |ctx: &mut SignatureTypeContext,
		                       parameters: &[ConstructorParameter]| {
			let param_types = {
				let mut typer = Typer::new(db, ctx, item, data);
				parameters
					.iter()
					.map(|p| {
						typer
							.complete_type(
								p.declared_type,
								None,
								TypeCompletionMode::EnumerationParameter,
							)
							.ty
					})
					.collect::<Box<[_]>>()
			};

			let mut had_error = false;
			for (p, t) in parameters.iter().zip(param_types.iter()) {
				if t.contains_error(db.upcast()) {
					had_error = true;
				}
				if !t.known_par(db.upcast()) || !t.known_enumerable(db.upcast()) {
					let (src, span) =
						NodeRef::from(EntityRef::new(db, item, p.declared_type)).source_span(db);
					ctx.add_diagnostic(
						item,
						TypeMismatch {
							src,
							span,
							msg: format!(
								"Expected par enumerable constructor parameter, but got '{}'",
								t.pretty_print(db.upcast())
							),
						},
					);
					had_error = true;
				}
			}

			(had_error, param_types)
		};

		for case in cases.iter() {
			match case {
				EnumConstructor::Named(Constructor::Atom { pattern }) => {
					self.add_declaration(PatternRef::new(item, *pattern), PatternTy::EnumAtom(ty));
				}
				EnumConstructor::Named(Constructor::Function {
					constructor,
					destructor,
					parameters,
				}) => {
					let (had_error, param_types) = get_param_types(self, parameters);
					let is_single = param_types.len() == 1;
					let mut constructors = Vec::with_capacity(6);
					let mut destructors = Vec::with_capacity(6);

					let mut add_ctor = |e: Ty, ps: Box<[Ty]>, l: bool| {
						destructors.push(FunctionEntry {
							has_body: false,
							overload: OverloadedFunction::Function(FunctionType {
								return_type: if is_single {
									ps[0]
								} else {
									Ty::tuple(db.upcast(), ps.iter().copied())
								},
								params: Box::new([e]),
							}),
						});
						constructors.push(EnumConstructorEntry {
							constructor: FunctionEntry {
								has_body: false,
								overload: OverloadedFunction::Function(FunctionType {
									return_type: e,
									params: ps,
								}),
							},
							is_lifted: l,
						});
					};

					// C(a, b, ..) -> E
					add_ctor(ty, param_types.clone(), false);
					if !had_error {
						// C(var a, var b, ..) -> var E
						add_ctor(
							ty.make_var(db.upcast()).unwrap(),
							param_types
								.iter()
								.map(|t| t.make_var(db.upcast()).unwrap())
								.collect::<Box<_>>(),
							false,
						);

						// C(opt a, opt b, ..) -> opt E
						add_ctor(
							ty.make_opt(db.upcast()),
							param_types
								.iter()
								.map(|t| t.make_opt(db.upcast()))
								.collect::<Box<_>>(),
							true,
						);
						// C(var opt a, var opt b, ..) -> var opt E
						add_ctor(
							ty.make_var(db.upcast()).unwrap().make_opt(db.upcast()),
							param_types
								.iter()
								.map(|t| t.make_var(db.upcast()).unwrap().make_opt(db.upcast()))
								.collect(),
							true,
						);
						// C(set of a, set of b, ..) -> set of E
						add_ctor(
							Ty::par_set(db.upcast(), ty).unwrap(),
							param_types
								.iter()
								.map(|t| Ty::par_set(db.upcast(), *t).unwrap())
								.collect(),
							true,
						);
						// C(var set of a, var set of b, ..) -> var set of E
						add_ctor(
							Ty::par_set(db.upcast(), ty)
								.unwrap()
								.make_var(db.upcast())
								.unwrap(),
							param_types
								.iter()
								.map(|t| {
									Ty::par_set(db.upcast(), *t)
										.unwrap()
										.make_var(db.upcast())
										.unwrap()
								})
								.collect(),
							true,
						);
					}

					self.add_declaration(
						PatternRef::new(item, *constructor),
						PatternTy::EnumConstructor(constructors.into_boxed_slice()),
					);
					self.add_declaration(
						PatternRef::new(item, *destructor),
						PatternTy::EnumDestructure(destructors.into_boxed_slice()),
					);
				}
				EnumConstructor::Anonymous {
					pattern,
					parameters,
				} => {
					let (_, param_tys) = get_param_types(self, parameters);
					self.add_declaration(
						PatternRef::new(item, *pattern),
						PatternTy::AnonymousEnumConstructor(Box::new(FunctionEntry {
							has_body: false,
							overload: OverloadedFunction::Function(FunctionType {
								return_type: ty,
								params: param_tys,
							}),
						})),
					);
				}
			}
		}
	}

	/// Get results of typing
	pub fn finish(self) -> (SignatureTypes, Vec<Error>) {
		(self.data, self.diagnostics)
	}
}

impl TypeContext for SignatureTypeContext {
	fn add_declaration(&mut self, pattern: PatternRef, declaration: PatternTy) {
		let old = self.data.patterns.insert(pattern, declaration);
		assert!(
			matches!(
				old,
				None | Some(PatternTy::Computing | PatternTy::ClassDecl { .. })
			),
			"Tried to add declaration for {:?} twice",
			pattern
		);
	}

	fn add_expression(&mut self, expression: ExpressionRef, ty: Ty) {
		let old = self.data.expressions.insert(expression, ty);
		assert!(
			old.is_none(),
			"Tried to add type for expression {:?} twice",
			expression
		);
	}

	fn add_identifier_resolution(&mut self, expression: ExpressionRef, resolution: PatternRef) {
		let old = self
			.data
			.identifier_resolution
			.insert(expression, resolution);
		assert!(
			old.is_none(),
			"Tried to add identifier resolution for {:?} twice",
			expression
		);
	}

	fn add_pattern_resolution(&mut self, pattern: PatternRef, resolution: PatternRef) {
		let old = self.data.pattern_resolution.insert(pattern, resolution);
		assert!(
			old.is_none(),
			"Tried to add pattern resolution for {:?} twice",
			pattern
		);
	}

	fn add_diagnostic(&mut self, item: ItemRef, e: impl Into<Error>) {
		// Suppress errors from other items
		if item == self.starting_item {
			self.diagnostics.push(e.into());
		}
	}

	fn add_type(&mut self, declared_ty: TypeRef, ty: Ty) {
		let old = self.data.types.insert(declared_ty, ty);
		assert!(
			old.is_none(),
			"Tried to add type for type {:?} twice",
			declared_ty
		);
	}

	fn get_type(&self, db: &dyn Hir, declared_ty: TypeRef) -> Ty {
		self.data.types[&declared_ty]
	}

	fn type_pattern(&mut self, db: &dyn Hir, pattern: PatternRef) -> PatternTy {
		// When computing signatures, we always type everything required
		// So other signatures get typed as well
		if let Some(d) = self.data.patterns.get(&pattern).cloned() {
			return d;
		}
		self.type_item(db, pattern.item());
		self.data.patterns[&pattern].clone()
	}
}
