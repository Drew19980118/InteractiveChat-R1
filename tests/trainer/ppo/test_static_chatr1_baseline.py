from verl.utils.reward_score.static_chatr1 import (
    static_chatr1_answer_references,
    static_chatr1_intent_rewards,
    static_chatr1_primary_answer_and_passage_ids,
    static_chatr1_rewrite,
)


def _inscit_candidates():
    return {
        "style": "rule",
        "ground_truth": [
            {
                "action": "answer",
                "response": "Paris is the capital of France.",
                "passage_id": ["p1"],
                "rewrite": "What is the capital of France?",
            },
            {
                "action": "answer",
                "response": "The capital city of France is Paris.",
                "passage_id": ["p2", "p1"],
                "rewrite": "What is the capital of France?",
            },
        ],
    }


def test_static_chatr1_preserves_multiple_references_and_gold_passage_union():
    ground_truth = _inscit_candidates()
    assert static_chatr1_answer_references(ground_truth) == [
        "Paris is the capital of France.",
        "The capital city of France is Paris.",
    ]
    assert static_chatr1_primary_answer_and_passage_ids(ground_truth) == (
        "Paris is the capital of France.",
        ["p1", "p2"],
    )


def test_static_chatr1_uses_the_human_rewrite_and_credits_only_best_query():
    ground_truth = _inscit_candidates()
    assert static_chatr1_rewrite(ground_truth) == "What is the capital of France?"
    rewards = static_chatr1_intent_rewards(
        ["France capital", "What is the capital of France?", "weather in Paris"],
        static_chatr1_rewrite(ground_truth),
    )
    assert rewards[0] == 0.0
    assert rewards[2] == 0.0
    assert rewards[1] == 1.0


def test_static_chatr1_missing_rewrite_does_not_create_an_intermediate_reward():
    assert static_chatr1_intent_rewards(["query one", "query two"], "") == [0.0, 0.0]
