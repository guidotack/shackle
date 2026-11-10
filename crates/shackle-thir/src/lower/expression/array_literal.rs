//! Lowering of array literals
//!

use shackle_ty::{Ty, TyData};

use crate::{
	lower::expression::{ExpressionCollector, alloc_expression},
	source::Origin,
	*,
};

impl<'db, 'a, 'b, 'c> ExpressionCollector<'db, 'a, 'b, 'c> {
	/// Collect 1D array literal of type `ty`
	pub(super) fn collect_array_literal(
		&mut self,
		al: &shackle_hir::ArrayLiteral<'db>,
		ty: Ty<'db>,
		origin: impl Into<Origin<'db>>,
	) -> Expression<'db> {
		let element = self.array_element_ty(ty);
		alloc_expression(
			ArrayLiteral(
				al.members
					.iter()
					.map(|m| self.collect_member(*m, element))
					.collect(),
			),
			self,
			origin,
		)
	}

	/// Collect 2D array literal of type `ty`
	pub(super) fn collect_array_literal_2d(
		&mut self,
		al: &shackle_hir::ArrayLiteral2D<'db>,
		ty: Ty<'db>,
		origin: impl Into<Origin<'db>>,
	) -> Expression<'db> {
		let origin = origin.into();
		let element = self.array_element_ty(ty);
		// Desugar 2D array literal into array2d call
		let mut idx_array = |dim: &shackle_hir::MaybeIndexSet<'db>| match dim {
			shackle_hir::MaybeIndexSet::Indexed(es) => alloc_expression(
				ArrayLiteral(es.iter().map(|e| self.collect_expression(*e)).collect()),
				self,
				origin,
			),
			shackle_hir::MaybeIndexSet::NonIndexed(c) => alloc_expression(
				LookupCall {
					function: self.parent.ids.functions.set2array.into(),
					arguments: vec![if *c > 0 {
						alloc_expression(
							LookupCall {
								function: self.parent.ids.functions.dot_dot.into(),
								arguments: vec![
									alloc_expression(IntegerLiteral(1), self, origin),
									alloc_expression(IntegerLiteral(*c as i64), self, origin),
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
				function: self.parent.ids.functions.mzn_array_2d_literal.into(),
				arguments: vec![
					rows,
					columns,
					alloc_expression(
						ArrayLiteral(
							al.members
								.iter()
								.map(|e| self.collect_member(*e, element))
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

	/// Collect indexed array literal of type `ty`
	pub(super) fn collect_indexed_array_literal(
		&mut self,
		al: &shackle_hir::IndexedArrayLiteral<'db>,
		ty: Ty<'db>,
		origin: impl Into<Origin<'db>>,
	) -> Expression<'db> {
		let origin = origin.into();
		let element = self.array_element_ty(ty);
		let index = match ty.lookup(self.parent.db) {
			TyData::Array { dim, .. } => Some(*dim),
			_ => None,
		};
		// Desugar indexed array literal into arrayNd call
		if al.indices.len() == 1 {
			alloc_expression(
				LookupCall {
					function: self.parent.ids.functions.mzn_start_indexed_array.into(),
					arguments: vec![
						self.collect_member(al.indices[0], index),
						alloc_expression(
							ArrayLiteral(
								al.members
									.iter()
									.map(|e| self.collect_member(*e, element))
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
		} else {
			alloc_expression(
				LookupCall {
					function: self.parent.ids.functions.mzn_indexed_array.into(),
					arguments: vec![alloc_expression(
						ArrayLiteral(
							al.indices
								.iter()
								.zip(al.members.iter())
								.map(|(i, e)| {
									alloc_expression(
										TupleLiteral(vec![
											self.collect_member(*i, index),
											self.collect_member(*e, element),
										]),
										self,
										origin,
									)
								})
								.collect(),
						),
						self,
						origin,
					)],
				},
				self,
				origin,
			)
		}
	}

	/// The element type of a set type, if `ty` is one
	pub(super) fn set_element_ty(&self, ty: Ty<'db>) -> Option<Ty<'db>> {
		match ty.lookup(self.parent.db) {
			TyData::Set(_, _, element) => Some(*element),
			_ => None,
		}
	}

	/// The element type of an array type, if `ty` is one
	pub(super) fn array_element_ty(&self, ty: Ty<'db>) -> Option<Ty<'db>> {
		match ty.lookup(self.parent.db) {
			TyData::Array { element, .. } => Some(*element),
			_ => None,
		}
	}

	/// Collect a member of a literal or comprehension whose type says it holds
	/// values of type `expected`, if known
	pub(super) fn collect_member(
		&mut self,
		idx: shackle_hir::ExpressionId<'db>,
		expected: Option<Ty<'db>>,
	) -> Expression<'db> {
		match expected {
			Some(expected) => self.collect_expression_upcast(idx, expected),
			None => self.collect_expression(idx),
		}
	}
}
