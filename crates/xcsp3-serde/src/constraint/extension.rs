use std::{fmt::Display, str::FromStr};

use serde::{Deserialize, Serialize};

use crate::{
	parser::{
		integer::{deserialize_int_exps, deserialize_int_tuples, serialize_int_tuples, IntExp},
		serialize_list,
	},
	IntVal, MetaInfo,
};

// TODO: Support for "smart" extension

#[derive(Clone, Debug, PartialEq, Hash, Deserialize, Serialize)]
#[serde(bound(deserialize = "Identifier: FromStr", serialize = "Identifier: Display"))]
pub struct Extension<Identifier = String> {
	#[serde(flatten)]
	pub info: MetaInfo<Identifier>,
	#[serde(
		alias = "$text",
		deserialize_with = "deserialize_int_exps",
		serialize_with = "serialize_list"
	)]
	pub list: Vec<IntExp<Identifier>>,
	#[serde(
		default,
		skip_serializing_if = "Vec::is_empty",
		deserialize_with = "deserialize_int_tuples",
		serialize_with = "serialize_int_tuples"
	)]
	pub supports: Vec<Vec<IntVal>>,
	#[serde(
		default,
		skip_serializing_if = "Vec::is_empty",
		deserialize_with = "deserialize_int_tuples",
		serialize_with = "serialize_int_tuples"
	)]
	pub conflicts: Vec<Vec<IntVal>>,
}
