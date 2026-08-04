from model.credit_memo import create_credit_memo

for symbol in ["BBY", "AMGN", "AAL"]:
    result = create_credit_memo(
        symbol=symbol,
        use_openai=False,
        context_mode="full",
        policy_mode="llm_evaluated",
        prompt_mode="tight",
    )

    print(
        symbol,
        result["experiment_config"]["architectural_coverage"]["coverage_pct"],
        result["experiment_config"]["input_data_coverage"]["coverage_pct"],
        result["memo_context"]["credit_request"],
    )