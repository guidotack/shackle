use std::str::FromStr;

use lsp_types::Uri;
use miette::{SourceCode, SpanContents};
use shackle_compiler::hir::{db::Hir, ids::NodeRef};

pub fn span_contents_to_range(r: &dyn SpanContents) -> lsp_types::Range {
	let mut range = lsp_types::Range::default();
	range.start.line = r.line() as u32;
	range.start.character = r.column() as u32;
	range.end.line = range.start.line;
	range.end.character = range.start.character;

	let mut iter = r.data().iter().copied().peekable();
	while let Some(char) = iter.next() {
		if matches!(char, b'\r' | b'\n') {
			range.end.line += 1;
			range.end.character = 0;
			if char == b'\r' {
				let _ = iter.next_if_eq(&b'\n');
			}
		} else {
			range.end.character += 1;
		}
	}
	range
}

pub fn node_ref_to_location<T: Into<NodeRef>>(
	db: &dyn Hir,
	node: T,
) -> Option<lsp_types::Location> {
	let (src, span) = node.into().source_span(db);
	let span_contents = src.read_span(&span, 0, 0).ok()?;
	let uri = Uri::from_str(src.path()?.as_os_str().to_str().unwrap())
		.ok()
		.or_else(|| Uri::from_str(&format!("file:///{}", src.path()?.to_string_lossy())).ok())?;
	let range = span_contents_to_range(&*span_contents);
	Some(lsp_types::Location { uri, range })
}
