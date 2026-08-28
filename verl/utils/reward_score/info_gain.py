from openai import OpenAI
import re
import difflib
import string
import json
import os
from .ground_truth import (
    select_answer_ground_truth,
    select_static_chatr1_answer_ground_truth_with_passage_ids,
    select_static_convagent_answer_ground_truth_with_passage_ids,
)

def check_tags_balance(solution_str: str, *, allow_clarify: bool = False, allow_search: bool = False) -> bool:
    """Check if tags are properly paired
    
    Args:
        solution_str: String to check
    
    Returns:
        bool: Whether all tags are properly paired
    """
    # Tag pairs to check
    tags_to_check = ['code', 'tool_call', 'think', 'answer', 'nonanswer']
    if allow_clarify:
        tags_to_check.append('clarify')
    if allow_search:
        tags_to_check.append('search')
    
    for tag in tags_to_check:
        # Count start and end tags
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"
        
        start_count = solution_str.count(start_tag)
        end_count = solution_str.count(end_tag)
        
        # If start and end tag counts don't match, return False
        if start_count != end_count:
            return False
            
        # Check tag nesting order (ensure end tag doesn't appear before start tag)
        last_pos = -1
        while True:
            start_pos = solution_str.find(start_tag, last_pos + 1)
            if start_pos == -1:
                break
                
            end_pos = solution_str.find(end_tag, start_pos)
            if end_pos == -1:
                return False
                
            last_pos = end_pos
            
    return True

def extract_terminal_action(
    solution_str: str,
    *,
    allow_clarify: bool = False,
    allow_search: bool = False,
) -> tuple[str, bool]:
    """Parse the final assistant action without scoring its answer text.

    ``answer`` and empty ``nonanswer`` are normally the only valid terminal
    actions. Static ConvAgent evaluation additionally permits a non-empty
    ``clarify`` terminal action.
    The parser intentionally examines only the final assistant turn, so prior
    tool calls cannot be mistaken for a terminal prediction.
    """
    if not check_tags_balance(
        solution_str,
        allow_clarify=allow_clarify,
        allow_search=allow_search,
    ):
        return "invalid", False

    final_turn = solution_str.rsplit("\n<|im_start|>assistant\n", 1)[-1]
    if "<think>" not in final_turn or "</think>" not in final_turn:
        return "invalid", False

    answer_matches = list(re.finditer(r"<answer>(.*?)</answer>", final_turn, re.DOTALL))
    nonanswer_matches = list(re.finditer(r"<nonanswer>(.*?)</nonanswer>", final_turn, re.DOTALL))
    clarify_matches = list(re.finditer(r"<clarify>(.*?)</clarify>", final_turn, re.DOTALL))
    if len(answer_matches) + len(nonanswer_matches) + len(clarify_matches) != 1:
        return "invalid", False
    if answer_matches:
        return "answer", True
    if clarify_matches:
        if not allow_clarify or not clarify_matches[0].group(1).strip():
            return "invalid", False
        return "clarify", True
    if nonanswer_matches[0].group(1).strip():
        return "invalid", False
    return "nonanswer", True


def preprocess_text(text: str) -> str:
    """Preprocess text for dataset scoring
    
    Processing steps:
    1. Convert to lowercase
    2. Remove punctuation (.,!?;:'"()[]{}...)
    3. Remove extra whitespace
    """
    # Replace punctuation with spaces
    for punct in string.punctuation:
        text = text.replace(punct, ' ')
    
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading and trailing whitespace
    text = text.strip()
    return text

def deal_multi_labels(ground_truth):
    for item in ground_truth:
        if item['label'].lower() == 'false':
            return 'false'
    return 'true'



def compute_f1(
    solution_str,
    ground_truth,
    data_source,
    val_type='f1',
    static_convagent_mode: bool = False,
    static_chatr1_mode: bool = False,
) -> float:
    if static_chatr1_mode:
        ground_truth = select_static_chatr1_answer_ground_truth_with_passage_ids(
            ground_truth, data_source=data_source
        )[0]
    elif static_convagent_mode:
        ground_truth = select_static_convagent_answer_ground_truth_with_passage_ids(
            ground_truth, data_source=data_source
        )[0]
    else:
        ground_truth = select_answer_ground_truth(ground_truth, data_source=data_source)
    # Empty targets are deliberately unsupervised: they contribute no F1/EM
    # or policy reward.
    if not ground_truth.strip():
        return 0.0
    if data_source in ['Factbench', 'politifact', 'liar2']:
        ground_truth = json.loads(ground_truth)
        ground_truth = deal_multi_labels(ground_truth)
    solution_str = solution_str.lower()
    ground_truth = ground_truth.lower()
    ground_truths = ground_truth.split("<|answer_split|>")
    terminal_action, format_valid = extract_terminal_action(
        solution_str,
        allow_clarify=static_convagent_mode,
        allow_search=static_convagent_mode or static_chatr1_mode,
    )
    if not format_valid:
        # ChatR1's published objective supplies answer F1 and intent F1 only;
        # malformed output therefore receives zero task reward rather than an
        # additional hidden format penalty.
        return 0.0 if (val_type == 'noformatf1' or static_chatr1_mode) else -2.0

    # A syntactically valid nonanswer action is not a formatting failure. It
    # receives zero textual F1; the independent action-reward channel handles
    # whether this was the correct terminal action for the sample.
    if terminal_action in {'nonanswer', 'clarify'}:
        return 0.0

    try:
        final_turn = solution_str.rsplit("\n<|im_start|>assistant\n", 1)[-1]
        answer_match = re.search(r'<answer>(.*?)</answer>', final_turn, re.DOTALL)
        if not answer_match:
            return 0.0 if (val_type == 'noformatf1' or static_chatr1_mode) else -2.0
        answer_content = preprocess_text(answer_match.group(1).strip())
    except Exception as e:
        print(f"Error extracting answer content: {e}")
        return 0.0 if (val_type == 'noformatf1' or static_chatr1_mode) else -2.0
    
    max_score = 0.0
    
    for gt in ground_truths:
        # Preprocess ground truth
        gt = preprocess_text(gt)

        if val_type == 'em':
            if gt == answer_content:
                return 1.0
        else:
            # Tokenize answer and reference answer
            pred_tokens = set(answer_content.split())
            gt_tokens = set(gt.split())
            
            if not gt_tokens:  # Avoid division by zero
                continue
            if not pred_tokens:
                continue
            
            # Calculate common tokens count
            common_tokens = pred_tokens & gt_tokens
            
            # Calculate precision and recall
            precision = len(common_tokens) / len(pred_tokens) if pred_tokens else 0
            recall = len(common_tokens) / len(gt_tokens) if gt_tokens else 0
            
            # Calculate F1 score
            if precision + recall > 0:  # Avoid division by zero
                f1 = 2 * (precision * recall) / (precision + recall)
                max_score = max(max_score, f1)
            
    return max_score

def _char_pos_to_token_idx(char_pos, offset_mapping):
    """
    Find the token index corresponding to a character position
    
    Args:
        char_pos: Character position
        offset_mapping: offset_mapping returned by tokenizer, format [(start, end), ...]
    
    Returns:
        Corresponding token index
    """
    for i, (start, end) in enumerate(offset_mapping):
        if start <= char_pos < end:
            return i
        if char_pos < start:
            # char_pos is in the gap before current token, return previous token
            return max(0, i - 1)
    # If out of range, return last token
    return len(offset_mapping) - 1


def compute_score(
    solution_str,
    ground_truth,
    data_source,
    val_type='f1',
    info_gain_reward=[],
    tokenizer=None,
    is_validation=False,
    static_convagent_mode: bool = False,
    static_chatr1_mode: bool = False,
):
    """
    Compute token-level reward scores
    
    Uses tokenizer's offset_mapping feature for precise token-character position mapping,
    avoiding index calculation errors caused by subword tokenization.
    
    Args:
        solution_str: Complete response string generated by model
        ground_truth: Reference answer
        data_source: Data source
        val_type: Validation type ('f1', 'em', 'noformatf1')
        info_gain_reward: Information gain reward list for each turn
        tokenizer: HuggingFace tokenizer
        is_validation: Whether in validation mode
    
    Returns:
        scores list or dict containing multiple metrics
    """
    if tokenizer is None:
        raise ValueError("tokenizer cannot be None")

    if static_chatr1_mode:
        ground_truth = select_static_chatr1_answer_ground_truth_with_passage_ids(
            ground_truth, data_source=data_source
        )[0]
    elif static_convagent_mode:
        ground_truth = select_static_convagent_answer_ground_truth_with_passage_ids(
            ground_truth, data_source=data_source
        )[0]
    else:
        ground_truth = select_answer_ground_truth(ground_truth, data_source=data_source)
        
    alpha = 1.0

    # Compute F1/EM scores
    if is_validation:
        f1_score = compute_f1(
            solution_str,
            ground_truth,
            data_source,
            val_type='f1',
            static_convagent_mode=static_convagent_mode,
            static_chatr1_mode=static_chatr1_mode,
        )
        em_score = compute_f1(
            solution_str,
            ground_truth,
            data_source,
            val_type='em',
            static_convagent_mode=static_convagent_mode,
            static_chatr1_mode=static_chatr1_mode,
        )
        noformatf1_score = compute_f1(
            solution_str,
            ground_truth,
            data_source,
            val_type='noformatf1',
            static_convagent_mode=static_convagent_mode,
            static_chatr1_mode=static_chatr1_mode,
        )
    else:
        f1_score = compute_f1(
            solution_str,
            ground_truth,
            data_source,
            val_type,
            static_convagent_mode=static_convagent_mode,
            static_chatr1_mode=static_chatr1_mode,
        )
    
    # Use offset_mapping to get precise token-character position mapping
    encoding = tokenizer(solution_str, return_offsets_mapping=True, add_special_tokens=False)
    token_ids = encoding['input_ids']
    offset_mapping = encoding['offset_mapping']  # [(char_start, char_end), ...]
    
    tokens_size = len(token_ids)
    scores = [0.0] * tokens_size

    # If no tokens, return empty scores directly
    if tokens_size == 0:
        if is_validation:
            return {"f1": f1_score, "em": em_score, "noformatf1": noformatf1_score, "scores": scores}
        return scores

    # Turn separator
    separator = "\n<|im_start|>assistant\n"
    
    # Find start character positions of all turns
    turn_start_positions = []  # Start character position of each turn's content
    turn_end_positions = []    # End character position of each turn's content
    
    # Find all separator positions
    sep_positions = []
    search_pos = 0
    while True:
        sep_pos = solution_str.find(separator, search_pos)
        if sep_pos == -1:
            break
        # 工具/检索文本中的同名字符串不能伪造 assistant turn。
        prefix = solution_str[:sep_pos].rstrip()
        if sep_pos == 0 or prefix.endswith("<|im_end|>"):
            sep_positions.append(sep_pos)
        search_pos = sep_pos + 1
    
    if len(sep_positions) == 0:
        # No separator, treat entire string as one turn
        turn_start_positions = [0]
        turn_end_positions = [len(solution_str)]
    else:
        # Content before first separator (if any)
        if sep_positions[0] > 0:
            turn_start_positions.append(0)
            turn_end_positions.append(sep_positions[0])
        
        # Turn after each separator
        for i, sep_pos in enumerate(sep_positions):
            turn_start = sep_pos + len(separator)
            turn_start_positions.append(turn_start)
            
            # Determine end position of this turn
            if i + 1 < len(sep_positions):
                turn_end = sep_positions[i + 1]
            else:
                turn_end = len(solution_str)
            turn_end_positions.append(turn_end)
    
    chats_size = len(turn_start_positions)

    # If no info_gain_reward or only one turn, put f1_score only on last token
    if info_gain_reward == [] or chats_size == 1:
        scores[-1] = alpha * f1_score
        
        if is_validation:
            return {"f1": f1_score, "em": em_score, "noformatf1": noformatf1_score, "scores": scores}
        return scores
    
    # Check if info_gain_reward length matches
    if len(info_gain_reward) != chats_size - 1:
        print(f"info_gain.py: turn mismatch - chats_size={chats_size}, info_gain_len={len(info_gain_reward)}")
        # When length doesn't match, fall back to using only f1_score
        scores[-1] = alpha * f1_score
        
        if is_validation:
            return {"f1": f1_score, "em": em_score, "noformatf1": noformatf1_score, "scores": scores}
        return scores

    # Assign reward to last token of each turn
    for i in range(chats_size):
        turn_end_char = turn_end_positions[i]
        
        # Find token index corresponding to last character of this turn
        # Note: turn_end_char is open interval, so use turn_end_char - 1
        if turn_end_char > 0:
            last_token_idx = _char_pos_to_token_idx(turn_end_char - 1, offset_mapping)
        else:
            last_token_idx = 0
        
        # Ensure index is within valid range
        last_token_idx = min(last_token_idx, tokens_size - 1)
        
        # Assign reward
        if i < chats_size - 1:
            # Additional boundary check to prevent IndexError
            if i < len(info_gain_reward):
                ig_value = info_gain_reward[i]
                if ig_value == 0.0:
                    ig_value = 1e-10  # Avoid being skipped by !=0 check
                scores[last_token_idx] = ig_value
            # else: skip this turn's info_gain (defensive, should not happen)
        else:
            scores[last_token_idx] = alpha * f1_score
    
    if is_validation:
        return {"f1": f1_score, "em": em_score, "noformatf1": noformatf1_score, "scores": scores}
    return scores
