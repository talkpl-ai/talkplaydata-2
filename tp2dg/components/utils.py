import concurrent.futures
import re

from tp2dg.entities.token_usage import TokenUsage


def call_with_timeout(func, timeout=120, *args, **kwargs):
	with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
		future = executor.submit(func, *args, **kwargs)
		try:
			return future.result(timeout=timeout)
		except concurrent.futures.TimeoutError as e:
			raise TimeoutError(f"API call timed out after {timeout} seconds") from e


def extract_detailed_token_usage(response) -> TokenUsage:
	try:
		if not response or not hasattr(response, "to_json_dict"):
			return TokenUsage()
		response_dict = response.to_json_dict()
		usage_metadata = response_dict.get("usage_metadata")
		if not usage_metadata:
			return TokenUsage()
		input_text_tokens = 0
		input_image_tokens = 0
		input_audio_tokens = 0
		prompt_details = usage_metadata.get("prompt_tokens_details", [])
		if not prompt_details:
			input_text_tokens = usage_metadata.get("prompt_token_count", 0)
		else:
			for item in prompt_details:
				modality = item.get("modality", "").lower()
				token_count = item.get("token_count", 0)
				if "audio" in modality:
					input_audio_tokens += token_count
				elif "image" in modality:
					input_image_tokens += token_count
				else:
					input_text_tokens += token_count
		response_details = usage_metadata.get("candidates_tokens_details", [])
		return TokenUsage(
			input_text_tokens=input_text_tokens,
			input_image_tokens=input_image_tokens,
			input_audio_tokens=input_audio_tokens,
			output_tokens=usage_metadata.get("candidates_token_count", 0),
			prompt_details=prompt_details,
			response_details=response_details,
		)
	except Exception:
		return TokenUsage()


def robust_parse_yaml_response(text: str, expected_keys: list[str]) -> dict[str, str]:
	result = {}
	clean_text = re.sub(r"```\w*\n?", "", text)
	clean_text = re.sub(r"\n```", "", clean_text)
	clean_text = clean_text.strip()
	key_positions = []
	for key in expected_keys:
		patterns = [rf"^{re.escape(key)}\s*:", rf"{re.escape(key)}\s*:"]
		for pattern in patterns:
			match = re.search(pattern, clean_text, re.MULTILINE)
			if match:
				key_positions.append((match.start(), key))
				break
	key_positions.sort(reverse=True)
	for i, (start_pos, key) in enumerate(key_positions):
		if i > 0:
			end_pos = key_positions[i - 1][0]
			section = clean_text[start_pos:end_pos]
		else:
			section = clean_text[start_pos:]
		key_pattern = rf"^{re.escape(key)}\s*:\s*(.+)"
		match = re.search(key_pattern, section, re.MULTILINE | re.DOTALL)
		if match:
			value = match.group(1).strip()
			value = re.sub(r'^["\']|["\']$', "", value)
			lines = value.split("\n")
			processed_lines = []
			for line in lines:
				line = line.strip()
				if line:
					processed_lines.append(line)
			value = " ".join(processed_lines)
			value = re.sub(r"\s+", " ", value).strip()
			if value:
				result[key] = value
	return result


def parse_recsys_choice_index(text: str, max_index: int) -> int:
	m = re.search(r"(?i)choice\s*[:=]\s*(\d+)", text)
	if m:
		idx = int(m.group(1))
		return min(max(idx, 0), max_index)
	m = re.search(r"(?i)index\s*(\d+)", text)
	if m:
		idx = int(m.group(1))
		return min(max(idx, 0), max_index)
	for line in text.splitlines():
		m = re.search(r"\b(\d+)\b", line)
		if m:
			idx = int(m.group(1))
			return min(max(idx, 0), max_index)
	return 0 