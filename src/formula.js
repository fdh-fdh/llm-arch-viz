// Zero-dependency math formula renderer (FR-E2).
// Input: a tiny AST; output: HTML string styled by .fm-* classes in styles.css.
// Node types:
//   'string'                 -> auto: single latin letters italic (variable), rest upright
//   { r: [nodes] }           -> horizontal row
//   { frac: [num, den] }     -> fraction
//   { sqrt: node }           -> square root
//   { sup: [base, exp] }     -> superscript      { sub: [base, sub] } -> subscript
//   { subsup: [base, sub, sup] }
//   { sum: { below, above, body } } -> big operator Σ
//   { func: [name, arg] }    -> upright function name + parenthesized arg
//   { paren: node }          -> ( node )
//   { text: 'str' }          -> force upright text
//   { v: 'str' }             -> force italic variable
//   { b: 'str' }             -> bold (matrices / vectors)

function esc(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function renderStr(s) {
  // auto-style: single latin letters italic; greek/digits/operators upright
  return String(s).split(/([a-zA-Z])/).map((part, i) =>
    i % 2 ? `<i>${part}</i>` : esc(part)
  ).join('');
}

export function fm(node) {
  if (node == null) return '';
  if (typeof node === 'string') return renderStr(node);
  if (Array.isArray(node)) return node.map(fm).join('');
  if (node.r) return node.r.map(fm).join('');
  if (node.text !== undefined) return `<span class="fm-up">${esc(node.text)}</span>`;
  if (node.v !== undefined) return `<i>${esc(node.v)}</i>`;
  if (node.b !== undefined) return `<b class="fm-b">${esc(node.b)}</b>`;
  if (node.frac) {
    return `<span class="fm-frac"><span class="fm-num">${fm(node.frac[0])}</span><span class="fm-den">${fm(node.frac[1])}</span></span>`;
  }
  if (node.sqrt) {
    return `<span class="fm-sqrt"><span class="fm-rad">√</span><span class="fm-radicand">${fm(node.sqrt)}</span></span>`;
  }
  if (node.sup) return `${fm(node.sup[0])}<sup>${fm(node.sup[1])}</sup>`;
  if (node.sub) return `${fm(node.sub[0])}<sub>${fm(node.sub[1])}</sub>`;
  if (node.subsup) return `${fm(node.subsup[0])}<span class="fm-stack"><sup>${fm(node.subsup[2])}</sup><sub>${fm(node.subsup[1])}</sub></span>`;
  if (node.sum) {
    return `<span class="fm-bigop"><span class="fm-above">${fm(node.sum.above || '')}</span><span class="fm-op">Σ</span><span class="fm-below">${fm(node.sum.below || '')}</span></span>${fm(node.sum.body)}`;
  }
  if (node.func) return `<span class="fm-up">${esc(node.func[0])}</span>(${fm(node.func[1])})`;
  if (node.paren) return `(${fm(node.paren)})`;
  return '';
}

// Render a display-style formula block.
export function formulaHTML(ast) {
  return `<div class="fm">${fm(ast)}</div>`;
}
