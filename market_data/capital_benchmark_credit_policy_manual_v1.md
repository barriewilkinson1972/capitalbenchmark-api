# Capital Benchmark Illustrative Corporate Credit Policy Manual

**Version:** v1.0  
**Scope:** Public-company corporate credit memo benchmark  
**Status:** Illustrative research policy, not a bank-specific policy manual

## 1. Purpose and Use

This policy manual defines minimum credit standards for illustrative corporate lending requests used in the Capital Benchmark research platform. It is not intended to represent any specific bank’s policy. Its purpose is to provide a controlled ground-truth reference for testing whether an AI-generated credit memo correctly identifies policy breaches, required mitigants, escalation requirements, and missing information.

The policy applies to corporate borrowers assessed using Capital Benchmark rating, PD, financial metrics, facility request details, and stress-testing outputs. AI-generated memos must not recommend approval where a request breaches policy unless the memo clearly identifies the breach, explains the required exception process, and states that approval is conditional on senior credit approval.

## 2. Minimum Credit Standards

| Policy ID | Requirement | Policy Rule |
|---|---|---|
| CP-01 | Minimum rating | New or increased unsecured corporate exposure requires a minimum Capital Benchmark rating of **BBB-**. Borrowers rated **BB+ or below** require exception approval. |
| CP-02 | Maximum leverage | For non-financial corporates, **Debt / EBITDA must not exceed 4.0x** for unsecured facilities unless specifically approved as an exception. |
| CP-03 | Severe leverage trigger | **Debt / EBITDA above 6.0x** is a high-severity policy breach and requires senior credit committee approval. |
| CP-04 | Net leverage | **Net Debt / EBITDA above 5.0x** requires enhanced review of liquidity, debt maturity profile, and refinancing capacity. |
| CP-05 | Profitability | Borrowers with **negative profit margin** require explanation of whether losses are temporary, recurring, accounting-driven, or structural. |
| CP-06 | Liquidity | **Cash / Debt below 10%** requires enhanced liquidity review and confirmation of committed funding sources. |
| CP-07 | Agency rating gap | If no agency rating is available, the memo must state that the CB rating cannot be externally benchmarked. |
| CP-08 | Data quality | Facilities must not be approved solely on model output where rating context quality is incomplete, stale, or scored using significant imputation. |
| CP-09 | Existing exposure | If existing exposure is not supplied, the memo must identify this as missing information before final credit decision. |
| CP-10 | Facility purpose | The memo must state the facility purpose. Vague purposes such as “general corporate purposes” require additional relationship-manager explanation for material increases. |

## 3. Facility Approval Guidance

### 3.1 Standard Approval Zone

A facility request may be treated as within standard policy parameters where all of the following are true:

- CB rating is BBB- or better.
- Debt / EBITDA is 4.0x or below.
- Net Debt / EBITDA is 5.0x or below.
- Profit margin is positive.
- Cash / Debt is at least 10%.
- Rating context quality is complete.
- No material adverse media flags are present.
- Existing and proposed exposure are both supplied.
- Facility purpose is sufficiently specific.

### 3.2 Enhanced Review Zone

A facility requires enhanced credit review where one or more of the following are true:

- Debt / EBITDA is above 4.0x but no higher than 6.0x.
- Net Debt / EBITDA is above 5.0x.
- Profit margin is negative.
- Cash / Debt is below 10%.
- No agency rating is available.
- Facility purpose is vague or unspecified.
- Existing exposure is missing.

In these cases, the memo must identify the relevant policy trigger, explain the credit concern, and list the additional information required before approval.

### 3.3 Exception Approval Zone

A facility requires exception approval where one or more of the following are true:

- CB rating is BB+ or below.
- Debt / EBITDA is above 6.0x.
- Rating context quality is materially incomplete.
- Material adverse media flags are present.
- Requested increase is large relative to existing exposure and no rationale is supplied.

Where exception approval is required, the memo must not present the request as suitable for ordinary approval. It must state that approval would require documented exception approval and senior credit sign-off.

## 4. Credit Memo Requirements

Every credit memo must include:

- Borrower and request summary
- Capital Benchmark rating assessment
- Financial risk assessment
- Policy compliance assessment
- Key credit strengths
- Key credit watchpoints
- Policy breaches
- Missing information
- Relationship-manager questions
- Credit committee focus areas
- Data quality and limitations

The memo must distinguish clearly between rating model drivers, financial watchpoints, policy breaches, missing information, and human judgement items.

The memo must not infer facts not supplied in the structured context. In particular, it must not claim market leadership, improving trends, covenant compliance, refinancing capacity, parent support, agency rating support, competitive strength, operational inefficiency, or approval recommendation unless those facts are explicitly supplied.

## 5. Policy Breach Language

Where a policy threshold is breached, the memo should use direct language.

**Example severe leverage breach:**  
“The proposed facility breaches CP-03 because Debt / EBITDA is 8.8x, above the 6.0x severe leverage trigger. The request would therefore require senior credit committee exception approval.”

**Example missing exposure:**  
“Existing exposure is not supplied. Under CP-09, the memo should identify this as missing information before a final credit decision.”

**Example missing agency rating:**  
“No agency rating is available in the supplied context. Under CP-07, the Capital Benchmark rating cannot be externally benchmarked and should be treated as a proprietary rating estimate.”

## 6. Benchmark Evaluation Rules

An LLM-generated credit memo should be marked down if it:

- Fails to identify a clear policy breach.
- Describes a breached facility as ordinary-course approval.
- Recommends approval without exception language.
- Confuses rating drivers with policy compliance.
- Fails to identify missing exposure, tenor, security, or purpose information.
- Invents mitigants not present in the supplied data.
- Claims agency rating support where no agency rating exists.
- Uses vague language where a hard threshold has been breached.

A memo should be marked positively if it:

- Identifies the correct policy IDs.
- Explains the policy breach using supplied numbers.
- Separates model output from credit judgement.
- Lists required mitigants or missing information.
- Uses conditional approval language appropriately.
- Escalates severe breaches to senior credit committee.

## 7. Illustrative Application

If a borrower has:

- CB rating: BBB+
- Debt / EBITDA: 8.8x
- Net Debt / EBITDA: 8.2x
- Profit margin: -0.07%
- Cash / Debt: 6.85%
- No agency rating
- Existing exposure: not supplied
- Facility purpose: general corporate purposes

Then the memo should identify:

- CP-03 severe leverage breach
- CP-04 net leverage enhanced review
- CP-05 negative profitability review
- CP-06 liquidity review
- CP-07 no agency rating comparison
- CP-09 missing existing exposure
- CP-10 vague facility purpose

The correct policy conclusion would be:

> The request is outside standard policy parameters and would require enhanced review and senior credit committee exception approval before any approval could be considered.
