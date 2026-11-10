//! The contribution registry and slot identity arithmetic.
//!
//! Each introduction of objects into a class is a *contribution* occupying a
//! contiguous block of that class's potential universe. This module records
//! contributions, tracks whether one leaves the class's defined fields
//! functionally determined, and converts between a contribution's local slot
//! ordinal and the global identity it maps to.

use shackle_hir::{Item, ids::PatternRef};
use shackle_ty::{Ty, TyData};

use crate::{lower::ItemCollector, *};

impl<'db> ItemCollector<'db> {
	pub(in crate::lower) fn register_class_object_contribution(
		&mut self,
		target_class: PatternRef<'db>,
		contribution_index: usize,
		declaration: DeclarationId<'db>,
		defined_fields_determined: bool,
	) {
		self.objects
			.class_object_contributions
			.entry(target_class)
			.or_default()
			.push((contribution_index, declaration));
		let _ = self
			.objects
			.class_contributions_all_determined
			.entry(target_class)
			.and_modify(|all| *all &= defined_fields_determined)
			.or_insert(defined_fields_determined);
		let _ = self.objects.contribution_determined_by_index.insert(
			(target_class, contribution_index),
			defined_fields_determined,
		);
	}

	/// The `defined_fields_determined` flag a specific contribution registered
	/// with (`None` when not yet registered). Projections reading every field
	/// from that contribution's decl inherit exactly this flag.
	pub(in crate::lower) fn contribution_determined(
		&self,
		target_class: PatternRef<'db>,
		contribution_index: usize,
	) -> Option<bool> {
		self.objects
			.contribution_determined_by_index
			.get(&(target_class, contribution_index))
			.copied()
	}

	pub(in crate::lower) fn class_object_contribution_declaration(
		&self,
		target_class: PatternRef<'db>,
		contribution_index: usize,
	) -> Option<DeclarationId<'db>> {
		self.objects
			.class_object_contributions
			.get(&target_class)
			.and_then(|contributions| {
				contributions.iter().find_map(|(index, declaration)| {
					(*index == contribution_index).then_some(*declaration)
				})
			})
	}

	/// The class identity of contribution storage slot `p` for
	/// `(class, contribution_index)`. Enum-indexed storage (nested
	/// per-contribution arrays) uses the slot index directly — it already IS
	/// the `<C>_occ_k(...)` identity. Int-indexed storage (top-level root
	/// contributions) is positioned from 1, so apply the contribution's enum
	/// constructor to the slot's argument (`slot_constructor_argument`) — the
	/// same `EnumMemberId::new(class_enum, contribution_index)` mapping the
	/// per-parent slice arrays use.
	pub(in crate::lower) fn contribution_slot_identity(
		&mut self,
		item: Item<'db>,
		class: PatternRef<'db>,
		contribution_index: usize,
		index_ty: Ty<'db>,
		index_expr: Expression<'db>,
	) -> Expression<'db> {
		if index_ty == Ty::par_int(self.db) {
			let enum_member = EnumMemberId::new(
				self.objects.class_map[&class].class_enum,
				contribution_index as u32,
			);
			let argument =
				self.slot_constructor_argument(item, class, contribution_index, index_expr);
			Expression::new(
				self.db,
				&self.model,
				item,
				Call {
					function: Callable::EnumConstructor(enum_member),
					arguments: vec![argument],
				},
			)
		} else {
			index_expr
		}
	}

	/// The constructor-LOCAL ordinal of contribution storage slot `p` for
	/// `(class, contribution_index)`: its position in the contribution's
	/// block, from 1. Int-indexed storage is positioned by
	/// local ordinals already. Enum-indexed storage erases to the class
	/// enum's global position, corrected back to constructor-local via the
	/// previous contribution's end offset — the same arithmetic as
	/// `project_class_identity` (falling back to the global position when no
	/// end offset was chained, matching the per-parent slice arithmetic).
	pub(in crate::lower) fn contribution_local_ordinal(
		&mut self,
		item: Item<'db>,
		class: PatternRef<'db>,
		contribution_index: usize,
		index_ty: Ty<'db>,
		index_expr: Expression<'db>,
	) -> Expression<'db> {
		if index_ty == Ty::par_int(self.db) {
			return index_expr;
		}
		let global_ordinal = Expression::new(
			self.db,
			&self.model,
			item,
			LookupCall {
				function: self.ids.functions.enum2int.into(),
				arguments: vec![index_expr],
			},
		);
		if contribution_index == 0 {
			return global_ordinal;
		}
		let Some(previous_end) = self
			.objects
			.contribution_end_map
			.get(&(class, contribution_index - 1))
			.copied()
		else {
			return global_ordinal;
		};
		let previous_end_expr = Expression::new(
			self.db,
			&self.model,
			item,
			ResolvedIdentifier::Declaration(previous_end),
		);
		let zero_based = Expression::new(
			self.db,
			&self.model,
			item,
			LookupCall {
				function: self.ids.functions.minus.into(),
				arguments: vec![global_ordinal, previous_end_expr],
			},
		);
		let one = Expression::new(self.db, &self.model, item, IntegerLiteral(1));
		Expression::new(
			self.db,
			&self.model,
			item,
			LookupCall {
				function: self.ids.functions.plus.into(),
				arguments: vec![one, zero_based],
			},
		)
	}

	/// The argument of the enum constructor of `(class, contribution_index)`
	/// for contribution storage slot `p`. It differs from the slot's
	/// constructor-local ordinal (`contribution_local_ordinal`) only for a
	/// top-level root whose block continues the numbering of the class's
	/// previous roots (`slot_constructor_argument`).
	pub(in crate::lower) fn contribution_slot_argument(
		&mut self,
		item: Item<'db>,
		class: PatternRef<'db>,
		contribution_index: usize,
		index_ty: Ty<'db>,
		index_expr: Expression<'db>,
	) -> Expression<'db> {
		if index_ty == Ty::par_int(self.db) {
			self.slot_constructor_argument(item, class, contribution_index, index_expr)
		} else {
			self.contribution_local_ordinal(item, class, contribution_index, index_ty, index_expr)
		}
	}

	/// The argument of the enum constructor for slot `p` of the int-indexed
	/// storage of `(class, contribution_index)`, a top-level root's: the
	/// storage is positioned from 1, but the constructor's domain continues
	/// the numbering of the class's previous roots (`<start>..<end> - 1`),
	/// so slot `p` is `<start> + p - 1`
	fn slot_constructor_argument(
		&mut self,
		item: Item<'db>,
		class: PatternRef<'db>,
		contribution_index: usize,
		index_expr: Expression<'db>,
	) -> Expression<'db> {
		let domain = self
			.class_enum_constructor_domain(
				self.objects.class_map[&class].class_enum,
				contribution_index,
			)
			.expect("object constructor has a bounded parameter domain");
		let start = match &*domain {
			ExpressionData::SetLiteral(members) if !members.is_empty() => members[0].clone(),
			ExpressionData::Call(Call {
				function: Callable::Function(f),
				arguments,
			}) if self.model[*f].name() == self.ids.functions.dot_dot => arguments[0].clone(),
			_ => Expression::new(
				self.db,
				&self.model,
				item,
				LookupCall {
					function: self.ids.functions.min.into(),
					arguments: vec![domain],
				},
			),
		};
		let offset = match &*start {
			ExpressionData::IntegerLiteral(IntegerLiteral(1)) => return index_expr,
			ExpressionData::IntegerLiteral(IntegerLiteral(start)) => {
				Expression::new(self.db, &self.model, item, IntegerLiteral(start - 1))
			}
			_ => {
				let one = Expression::new(self.db, &self.model, item, IntegerLiteral(1));
				Expression::new(
					self.db,
					&self.model,
					item,
					LookupCall {
						function: self.ids.functions.minus.into(),
						arguments: vec![start, one],
					},
				)
			}
		};
		Expression::new(
			self.db,
			&self.model,
			item,
			LookupCall {
				function: self.ids.functions.plus.into(),
				arguments: vec![index_expr, offset],
			},
		)
	}

	pub(in crate::lower) fn register_class_set_top_level_contribution(
		&mut self,
		target_class: PatternRef<'db>,
		contribution_index: usize,
		expression: Expression<'db>,
	) {
		self.objects
			.class_set_top_level_contributions
			.entry(target_class)
			.or_default()
			.push((contribution_index, expression));
	}

	pub(in crate::lower) fn record_expr_has_field(
		&self,
		expr: &Expression<'db>,
		field_ident: Identifier<'db>,
	) -> bool {
		matches!(
			expr.ty().lookup(self.db),
			TyData::Record(_, fields) if fields.iter().any(|(field, _)| *field == field_ident.0)
		)
	}
}
