//! Class-typed values in lowered expressions.
//!
//! A class-typed value denotes an identity in its class's potential universe,
//! the `<C>_potential` enum. These helpers relabel class types to those enums
//! and check that a lowered expression's type still matches what the
//! typechecker inferred. Converting an object of a subclass to its superclass
//! is in `upcast.rs`.

use shackle_hir::class_analysis::class_pattern_for;
use shackle_ty::{Ty, TyData};

use crate::{
	lower::expression::{ExpressionCollector, alloc_expression},
	source::Origin,
	*,
};

impl<'db, 'a, 'b, 'c> ExpressionCollector<'db, 'a, 'b, 'c> {
	/// Whether `actual` is a valid lowered form of the typechecker's
	/// `expected` type: the lowering substitutes `Class<C>` with the
	/// `<C>_potential` enum (and par-ifies singular fresh identities), so the
	/// constructed expression's type legitimately differs from the HIR type
	/// at exactly those points.
	pub(super) fn lowered_ty_matches(&self, actual: Ty<'db>, expected: Ty<'db>) -> bool {
		if actual == expected {
			return true;
		}
		let class_pattern = |class: shackle_ty::ClassRef<'db>| {
			class_pattern_for(self.parent.db, class).expect("class item for class type")
		};
		match (
			actual.lookup(self.parent.db),
			expected.lookup(self.parent.db),
		) {
			(
				TyData::Class(actual_inst, actual_opt, actual_class),
				TyData::Class(expected_inst, expected_opt, expected_class),
			) if actual_opt == expected_opt
				&& class_pattern(*actual_class) == class_pattern(*expected_class)
				&& (actual_inst == expected_inst
					|| (*actual_inst == VarType::Par && *expected_inst == VarType::Var)) =>
			{
				// par-class is a valid lowering of var-class: the identity is
				// par because the singular fresh introduction collapses it,
				// while the HIR type kept var to drive attribute varification
				// through the field-access cascade.
				true
			}
			(
				TyData::Set(actual_inst, actual_opt, actual_element),
				TyData::Set(expected_inst, expected_opt, expected_element),
			) if actual_inst == expected_inst
				&& actual_opt == expected_opt
				&& self.lowered_ty_matches(*actual_element, *expected_element) =>
			{
				true
			}
			(
				TyData::Array {
					opt: actual_opt,
					dim: actual_dim,
					element: actual_element,
				},
				TyData::Array {
					opt: expected_opt,
					dim: expected_dim,
					element: expected_element,
				},
			) if actual_opt == expected_opt
				&& self.lowered_ty_matches(*actual_dim, *expected_dim)
				&& self.lowered_ty_matches(*actual_element, *expected_element) =>
			{
				// An `array [..] of <C>` attribute lowers its element to the
				// substituted potential enum (`array [..] of var <C>_potential`),
				// so a sibling/field read of the whole array carries the enum
				// element while the HIR keeps the class element. Recurse on the
				// element exactly as the Set arm does — the class/enum element
				// arms below absorb the `<C>_potential`↔`Class<C>` equivalence.
				// The dimension recurses the same way: a class-indexed array
				// (`array [A] of int`) lowers its index type to `A_potential`.
				true
			}
			(
				TyData::Tuple(actual_opt, actual_fields),
				TyData::Tuple(expected_opt, expected_fields),
			) if actual_opt == expected_opt
				&& actual_fields.len() == expected_fields.len()
				&& actual_fields
					.iter()
					.zip(expected_fields.iter())
					.all(|(a, e)| self.lowered_ty_matches(*a, *e)) =>
			{
				// The dimensions of a multi-dimensional array, one of which may
				// be a class (`array [A, 1..2] of int`).
				true
			}
			(
				TyData::Enum(actual_inst, actual_opt, actual_enum),
				TyData::Class(expected_inst, expected_opt, expected_class),
			) if actual_opt == expected_opt
				&& (actual_inst == expected_inst
					|| (*actual_inst == VarType::Par && *expected_inst == VarType::Var)) =>
			{
				// `<C>_potential` is the lowered form of `Class<C>`, with the
				// same par-for-var allowance as the Class/Class arm above (a
				// singular fresh introduction collapses the identity to par).
				//
				// Matched only at the SAME level: a set/array wrapper is the
				// business of the arms above, which recurse into the element and
				// compare inst and opt as they go.
				self.parent.model
					[self.parent.objects.class_map[&class_pattern(*expected_class)].class_enum]
					.enum_type()
					== *actual_enum
			}
			(
				TyData::Set(actual_inst, actual_opt, actual_element),
				TyData::Set(expected_inst, expected_opt, expected_element),
			) if actual_inst == expected_inst && actual_opt == expected_opt => {
				let Some(actual_enum) = actual_element.enum_ty(self.parent.db) else {
					return false;
				};
				let Some(expected_class) = expected_element.class_type(self.parent.db) else {
					return false;
				};
				self.parent.model
					[self.parent.objects.class_map[&class_pattern(expected_class)].class_enum]
					.enum_type()
					== actual_enum
			}
			_ => false,
		}
	}

	/// Wrap an output-context expression in `fix` when the lowering produced a
	/// var value for a type the typer par-ified.
	///
	/// In an output context the typer pars both references to outside
	/// declarations and projections out of class storage, because output is
	/// evaluated against the solved model. The declaration or storage behind
	/// them is still var, so the lowered expression has to be fixed for HIR and
	/// THIR to agree on the inst. Returns `None` when this is not that
	/// situation — not an output context, the types already agree, or they
	/// differ by more than var/par — leaving the caller to handle it.
	pub(super) fn fix_for_output(
		&mut self,
		expr: &Expression<'db>,
		ty: Ty<'db>,
		origin: impl Into<Origin<'db>>,
	) -> Option<Expression<'db>> {
		let db = self.parent.db;
		if !self.in_output
			|| self.lowered_ty_matches(expr.ty(), ty)
			|| !self.lowered_ty_matches(expr.ty().make_par(db), ty)
		{
			return None;
		}
		let fixed = alloc_expression(
			LookupCall {
				function: self.parent.ids.functions.fix.into(),
				arguments: vec![expr.clone()],
			},
			self,
			origin,
		);
		assert!(
			self.lowered_ty_matches(fixed.ty(), ty),
			"output fix expected {} but produced {}",
			ty.pretty_print(db),
			fixed.ty().pretty_print(db),
		);
		Some(fixed)
	}

	/// Whether `actual` is the same lowered shape as `expected` modulo inst
	/// and opt at every level (and the class/potential-enum identification):
	/// the loosest form of the postcondition, for shapes where a var-set
	/// comprehension lift or storage varification legitimately changed both.
	pub(super) fn lowered_shape_matches(&self, actual: Ty<'db>, expected: Ty<'db>) -> bool {
		let db = self.parent.db;
		let class_enum = |class: shackle_ty::ClassRef<'db>| {
			class_pattern_for(db, class)
				.and_then(|p| self.parent.objects.class_map.get(&p))
				.map(|info| self.parent.model[info.class_enum].enum_type())
		};
		match (actual.lookup(db), expected.lookup(db)) {
			(TyData::Boolean(_, _), TyData::Boolean(_, _))
			| (TyData::Integer(_, _), TyData::Integer(_, _))
			| (TyData::Float(_, _), TyData::Float(_, _))
			| (TyData::String(_), TyData::String(_))
			| (TyData::Bottom(_), TyData::Bottom(_)) => true,
			(TyData::Enum(_, _, a), TyData::Enum(_, _, e)) => a == e,
			(TyData::Class(_, _, a), TyData::Class(_, _, e)) => {
				class_pattern_for(db, *a) == class_pattern_for(db, *e)
			}
			(TyData::Enum(_, _, a), TyData::Class(_, _, e)) => class_enum(*e) == Some(*a),
			(TyData::Class(_, _, a), TyData::Enum(_, _, e)) => class_enum(*a) == Some(*e),
			(TyData::Set(_, _, a), TyData::Set(_, _, e)) => self.lowered_shape_matches(*a, *e),
			(
				TyData::Array {
					dim: ad,
					element: ae,
					..
				},
				TyData::Array {
					dim: ed,
					element: ee,
					..
				},
			) => self.lowered_shape_matches(*ad, *ed) && self.lowered_shape_matches(*ae, *ee),
			(TyData::Tuple(_, afs), TyData::Tuple(_, efs)) => {
				afs.len() == efs.len()
					&& afs
						.iter()
						.zip(efs.iter())
						.all(|(a, e)| self.lowered_shape_matches(*a, *e))
			}
			(TyData::Record(_, afs), TyData::Record(_, efs)) => {
				afs.len() == efs.len()
					&& afs
						.iter()
						.zip(efs.iter())
						.all(|((an, a), (en, e))| an == en && self.lowered_shape_matches(*a, *e))
			}
			_ => false,
		}
	}

	/// Relabel a class-labeled call operand to its potential-enum lowering.
	///
	/// Function resolution and type specialisation instantiate generic
	/// parameters from the argument types, and the standard library is typed
	/// over enums — a `var Class<B>` operand meeting a `var set of
	/// B_potential` operand fails to instantiate `in(var $$E, var set of
	/// $$E)`. The runtime value of a class-labeled expression already IS the
	/// potential-enum identity, so the relabel is cosmetic and makes every
	/// call see consistent enum labels.
	pub(super) fn relabel_class_operand(&mut self, expr: Expression<'db>) -> Expression<'db> {
		if !expr
			.ty()
			.walk(self.parent.db)
			.any(|t| t.class_type(self.parent.db).is_some())
		{
			return expr;
		}
		let enum_ty = self.parent.substitute_class_with_potential_enum(expr.ty());
		if enum_ty == expr.ty() {
			return expr;
		}
		let mut relabeled = Expression::new_unchecked(enum_ty, (*expr).clone(), expr.origin());
		relabeled
			.annotations_mut()
			.extend(expr.annotations().iter().cloned());
		relabeled
	}
}
