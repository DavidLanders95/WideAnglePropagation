---
applyTo: "**/*.tex"
---

# Scientific Writing Voice

Write like the user: careful, direct, technically grounded, and explanatory. Preserve the scientific meaning, the author's cautious stance, and the local structure of the argument unless asked for a larger rewrite.

## Core Workflow

1. Read the relevant manuscript section before editing.
2. Identify the role of the text: motivation, method derivation, comparison, results interpretation, limitation, or conclusion.
3. Keep edits minimal by default: improve clarity, flow, grammar, and claim calibration without changing the technical content.
4. Prefer explicit contrasts between methods, approximations, and regimes.
5. Keep LaTeX commands, labels, citations, equations, and notation intact.
6. If a claim sounds too broad, narrow it by adding the regime, reference condition, or approximation responsible.

## Voice Rules

- Prefer direct topic sentences followed by the technical reason.
- Use cautious but clear claims: "indicates", "suggests", "appears", "in these cases", "under the conditions considered here", "to our knowledge".
- Make approximations explicit: distinguish split-step error, paraxial error, free-space propagation, local wavelength changes, backscattering, and reference-model assumptions.
- Compare methods by what changes physically and computationally, not by vague quality terms.
- When discussing results, state what the figure shows, then interpret what causes the difference, then state the practical implication.
- Keep novelty claims modest unless the evidence is strong.
- Avoid marketing language: "powerful", "breakthrough", "state-of-the-art", "highly robust", "dramatically".
- Avoid over-compression. Use explanatory bridges so the reader can follow why a method or conclusion follows.

## Preferred Sentence Moves

- "The main approximation that remains is..."
- "This is useful because..."
- "This distinction matters because..."
- "The change is not ..., but ..."
- "Apart from ..., the algorithm is ..."
- "At least in this case, ..."
- "Taken together, these results indicate ..."
- "In practice, this means that ..."
- "The balance between ... may change ..."

## Section-Specific Guidance

- **Abstracts**: state the standard method, its limitation, the proposed comparison or adaptation, the reference benchmark, and the main practical conclusion.
- **Introductions**: move from broad use case to why model accuracy matters, then to the standard method, its limitation, prior alternatives, and the specific gap addressed.
- **Methods**: define the governing equation or approximation first, then explain what each algorithm changes relative to the previous method.
- **Results**: avoid long single paragraphs. For each experiment, separate setup, visual observation, quantitative result, and interpretation.
- **Conclusions**: summarize the benchmark, identify the dominant source of error, state the practical recommendation, then give limitations and future work.
- **Captions**: describe what is shown, what is compared, and what quantity or normalization is used. Avoid unsupported claims in captions.

## Style Guide

### Default Tone

The preferred voice is rigorous but not ornate. Explains why a distinction matters and avoids pretending a method is better in all regimes. Comfortable with narrowing scope: "under the conditions considered here", "in this case".

The writing should feel like a careful scientist walking through a comparison: what the standard method does, what approximation limits it, what the proposed method changes, what the benchmark shows, what practical recommendation follows.

### Argument Structure

Prefer cumulative reasoning:
1. Start from the accepted baseline.
2. Name the specific limitation.
3. Introduce the smallest change or alternative.
4. Explain the physical or computational consequence.
5. Interpret the result only within the tested regime.

Example:
> The angular spectrum method keeps the same split-step structure but replaces the Fresnel kernel by the exact free-space dispersion relation. Apart from the change in the Fourier-space phase factor, the algorithm is the same as standard multislice and has the same per-slice cost. The approximation that remains is the split-step treatment of the specimen.

### Claim Calibration

Use strong language for directly supported observations:
- "F-MS diverges from the reference after..."
- "AS-MS and WP-MS follow the ODE reference closely..."
- "The difference is small and becomes apparent only at..."

Use careful language for interpretation:
- "This suggests that..."
- "These results indicate that..."
- "At least in this case..."
- "The balance may change at..."

Avoid unsupported universals:
- Do not write "WP-MS is superior" when the data only show small improvements in specific cases.
- Do not write "AS-MS solves high-angle scattering" when split-step and backscattering approximations remain.
- Do not write "no additional cost" unless the implementation context is clear; prefer "no additional per-slice FFT cost" or "almost no additional computational cost relative to F-MS".

### Syntax Preferences

- Use "therefore", "however", "although", "in particular", and "taken together" when they clarify the logic.
- Use appositives and parenthetical clarifications for method names and abbreviations.
- Prefer "we refer to..." when introducing a naming convention.
- Prefer "we use..." and "we compare..." in methods and results.
- Use "the method" or the abbreviation consistently after introducing a method.

### Common Repairs

- Split very long result paragraphs into smaller paragraphs organized by setup, observation, and implication.
- Replace "ground truth" with "reference" or "ODE reference" when a numerical benchmark is model-dependent.
- Replace "accurate tracking" with a measured quantity when one is available.
- Replace "visually worse" with the visible feature or error metric.
- Preserve useful authorial hedging, but remove repeated hedges in the same paragraph.

### LaTeX Handling

- Preserve labels, citations, equation references, and math notation.
- Keep nonbreaking spaces in units and crystallographic directions where appropriate: `100~mrad`, `40~nm`, `Au~[100]`.
- Prefer `Equation~\eqref{...}` and `Figure~\ref{...}` for formal references.
- Do not rewrite technical notation unless the manuscript is inconsistent or the user asks.

## Examples and Templates

### Result Paragraph Pattern
> Figure~\ref{...} shows [quantity] for [methods] in [test case]. The main difference is that [baseline method] [observed failure], while [alternative methods] [observed behavior] relative to the [reference]. This indicates that [specific approximation] is the dominant source of error in this regime. [Practical implication], although [limitation or remaining caveat].

### Method Comparison Pattern
> [Method B] keeps [shared structure] but replaces [specific approximation] with [more complete term]. Apart from this change, the algorithm remains [computational structure]. The approximation that remains is [remaining limitation].

### Cautious Recommendation Pattern
> For the [test regime] cases considered here, [method] provides most of the accuracy improvement over [baseline] while preserving [practical advantage]. [More complex method] remains more accurate, but its advantage is small in these benchmarks and comes at [cost]. The balance may change in [excluded regimes].

### Before/After: Overlong Result Sentence
Before:
> It is clear in Figure~\ref{...}, which displays the exit-wave intensity, phase, and image intensity, that the Fresnel method is qualitatively different from the other three methods.

After:
> Figure~\ref{...} compares the exit-wave intensity, phase, and image-plane intensity for the Au~[100] verification case. The Fresnel result is qualitatively different from AS-MS, WP-MS, and the Klein--Gordon ODE reference.

### Before/After: Overstated Practical Claim
Before:
> AS-MS is the best choice for all high-angle scattering simulations in TEM.

After:
> Under the TEM conditions tested here, AS-MS appears to be the most practical correction to F-MS because it captures most of the wide-angle improvement without changing the split-step structure or per-slice FFT cost.

### Before/After: Blurry Causal Explanation
Before:
> The F-MS method is not capable of tracking these beams, and thus its accuracy suffers.

After:
> F-MS cannot represent these high-angle components with the correct longitudinal phase, so their later contribution to the central beam is misestimated.

### Useful Sentence Openers
- "This form is useful because..."
- "The approximation that remains is..."
- "This should be distinguished from..."
- "The change is not the underlying idea, but..."
- "A practical way to mitigate this error is..."
- "The present benchmarks are restricted to..."
- "Future work will test whether..."
