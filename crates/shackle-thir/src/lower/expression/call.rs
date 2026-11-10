//! Lowering of calls to THIR
//!
//! This needs special handling for objects passed where a superclass is
//! expected, and for `show` of objects

use rustc_hash::FxHashMap;
use shackle_hir::ids::ExpressionRef;

use crate::{
	lower::{
		LoweredIdentifier,
		expression::{ExpressionCollector, alloc_expression},
	},
	*,
};

impl<'db, 'a, 'b, 'c> ExpressionCollector<'db, 'a, 'b, 'c> {
	/// Lower a call. Each argument is converted to the parameter type of the
	/// overload the type checker chose (see `upcast`).
	pub(super) fn collect_call_expression(
		&mut self,
		c: &shackle_hir::Call<'db>,
		idx: shackle_hir::ExpressionId<'db>,
	) -> Expression<'db> {
		let db = self.parent.db;
		let origin = ExpressionRef::new(db, self.item, idx).into_entity(db);
		let function = if let shackle_hir::Expression::Identifier(_) = self.data[c.function] {
			let res = self.types.name_resolution(c.function).unwrap_or_else(|| {
				panic!(
					"No name resolution in types for {:?} at {:?}",
					c.function,
					ExpressionRef::new(self.parent.db, self.item, c.function)
						.source_span(self.parent.db)
				);
			});
			let ident = self.parent.resolutions.get(&res).unwrap_or_else(|| {
				let f = ExpressionRef::new(self.parent.db, self.item, c.function);
				panic!(
					"Did not lower {:?} at {:?} used by {:?} at {:?}",
					res,
					res.into_entity(self.parent.db).source_span(self.parent.db),
					f,
					f.source_span(self.parent.db),
				)
			});
			match ident {
				LoweredIdentifier::Callable(c) => c.clone(),
				_ => Callable::Expression(Box::new(self.collect_expression(c.function))),
			}
		} else {
			Callable::Expression(Box::new(self.collect_expression(c.function)))
		};
		// Each argument is converted to the parameter type of the overload
		// the type checker chose, so an object of a subclass passed where its
		// superclass is expected becomes the same object of the superclass
		// (see `upcast`). Class-typed arguments then take their potential
		// enum types, which function resolution and type specialisation
		// instantiate the standard library's enum parameters from.
		let params = self
			.types
			.get_expression(c.function)
			.and_then(|f| f.function_params(db));
		let mut arguments = c
			.arguments
			.iter()
			.enumerate()
			.map(|(i, arg)| {
				let expr = match params.as_ref().and_then(|params| params.get(i)) {
					Some(param) => self.collect_expression_upcast(*arg, *param),
					None => self.collect_expression(*arg),
				};
				self.relabel_class_operand(expr)
			})
			.collect::<Vec<_>>();

		// `show` of an object reference prints the object's id, so show the
		// ids in place of the identities (this also covers string
		// interpolation, which desugars to `show`). The changed argument type
		// makes the call re-dispatch below.
		let is_show = matches!(
			&self.data[c.function],
			shackle_hir::Expression::Identifier(id)
				if *id == self.parent.ids.functions.show
					|| *id == self.parent.ids.functions.show_json
		);
		if is_show && let [argument] = arguments.as_mut_slice() {
			// A class-indexed array is shown keyed by the ids, which is not
			// the `show` of any MiniZinc value, so the whole call is replaced.
			let json = matches!(
				&self.data[c.function],
				shackle_hir::Expression::Identifier(id)
					if *id == self.parent.ids.functions.show_json
			);
			if let Some(shown) = self
				.parent
				.show_class_indexed_array(origin.into(), argument, json)
			{
				return shown;
			}
			if let Some(with_ids) = self.parent.show_argument_with_ids(argument) {
				*argument = with_ids;
			}
		}

		let params = match &function {
			Callable::Function(f) => Some(self.parent.model[*f].parameters()),
			Callable::Annotation(a) => self.parent.model[*a].parameters.as_ref().map(|v| &v[..]),
			Callable::EnumConstructor(e) => self.parent.model[e.enumeration_id()]
				.definition()
				.unwrap()[e.member_index() as usize]
				.parameters
				.as_ref()
				.map(|v| &v[..]),
			_ => None,
		};

		if let Some(params) = params
			&& params.len() > arguments.len()
		{
			// Need to fill in default and named arguments
			let params = params[arguments.len()..].to_vec();
			let mut named = c
				.named_arguments
				.iter()
				.map(|(name, arg)| {
					(
						self.data[*name].identifier().unwrap(),
						self.collect_expression(*arg),
					)
				})
				.collect::<FxHashMap<_, _>>();

			for param in params {
				let param_name = self.parent.model[param].name().unwrap();
				if let Some(arg) = named.remove(&param_name) {
					arguments.push(arg);
				} else {
					let default = self.parent.param_defaults[&param].clone();
					arguments.push(default);
				}
			}
		}

		// The HIR-resolved function item may no longer match the
		// lowered argument types: varified storage widens par HIR
		// operands to var, and class operands are relabeled to their
		// potential enums. Re-dispatch by name so the call binds the
		// overload for the actual argument types (a `LookupCall`
		// resolves straight back to a `Call`, so a still-matching
		// resolution is unchanged).
		let needs_redispatch = match &function {
			Callable::Function(f) => {
				let params = self.parent.model[*f].parameters();
				params.len() != arguments.len()
					|| !arguments
						.iter()
						.zip(params.iter())
						.all(|(arg, p)| arg.ty().is_subtype_of(db, self.parent.model[*p].ty()))
			}
			_ => false,
		};
		if needs_redispatch {
			let Callable::Function(f) = &function else {
				unreachable!()
			};
			let name = self.parent.model[*f].name();
			alloc_expression(
				LookupCall {
					function: name,
					arguments,
				},
				self,
				origin,
			)
		} else {
			alloc_expression(
				Call {
					function,
					arguments,
				},
				self,
				origin,
			)
		}
	}
}
