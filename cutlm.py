import hashlib
import hmac
import base64
import datetime
import json
import requests
import re
from tqdm import tqdm
from difflib import SequenceMatcher

class GetResult:
    def __init__(self, host):
        self.APPID = ""
        self.Secret = ""
        self.APIKey = ""
        self.Host = host
        self.RequestUri = "/v2/ots"
        self.url = "https://" + host + self.RequestUri
        self.HttpMethod = "POST"
        self.Algorithm = "hmac-sha256"
        self.HttpProto = "HTTP/1.1"
        self.Date = self.httpdate(datetime.datetime.utcnow())
        self.BusinessArgs = {"from": "zh", "to": "ur"}

    def hashlib_256(self, res):
        m = hashlib.sha256(res.encode('utf-8')).digest()
        result = "SHA-256=" + base64.b64encode(m).decode('utf-8')
        return result

    def httpdate(self, dt):
        weekday = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][dt.weekday()]
        month = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][dt.month - 1]
        return "%s, %02d %s %04d %02d:%02d:%02d GMT" % (weekday, dt.day, month, dt.year, dt.hour, dt.minute, dt.second)

    def generateSignature(self, digest):
        signatureStr = "host: " + self.Host + "\n"
        signatureStr += "date: " + self.Date + "\n"
        signatureStr += self.HttpMethod + " " + self.RequestUri + " " + self.HttpProto + "\n"
        signatureStr += "digest: " + digest
        signature = hmac.new(self.Secret.encode('utf-8'),
                             signatureStr.encode('utf-8'),
                             digestmod=hashlib.sha256).digest()
        result = base64.b64encode(signature)
        return result.decode('utf-8')

    def init_header(self, data):
        digest = self.hashlib_256(data)
        sign = self.generateSignature(digest)
        authHeader = 'api_key="%s", algorithm="%s", ' \
                     'headers="host date request-line digest", ' \
                     'signature="%s"' % (self.APIKey, self.Algorithm, sign)
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Host": self.Host,
            "Date": self.Date,
            "Digest": digest,
            "Authorization": authHeader
        }
        return headers

    def get_body(self, text):
        content = base64.b64encode(text.encode('utf-8')).decode('utf-8')
        postdata = {
            "common": {"app_id": self.APPID},
            "business": self.BusinessArgs,
            "data": {"text": content}
        }
        body = json.dumps(postdata)
        return body

    def call_url(self, text):
        self.Date = self.httpdate(datetime.datetime.utcnow())

        body = self.get_body(text)
        headers = self.init_header(body)

        response = requests.post(self.url, data=body, headers=headers, timeout=8)
        if response.status_code == 200:
            respData = json.loads(response.text)
            if respData["code"] == 0:
                translated_text = respData["data"]["result"]["trans_result"]["dst"]
                return translated_text
            else:
                print(f"翻译失败，错误码：{respData['code']}，错误信息：{respData.get('message', '')}")
        else:
            print(f"HTTP请求失败，状态码：{response.status_code}，错误信息：{response.text}")
        return None

def read_bio_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read().strip().split('\n\n')
    return data

def write_translate_data(file_path, data):
    with open(file_path, 'a', encoding='utf-8') as file:
        file.write('\n\n'.join(data) + '\n\n')

# Function to mark entities in the text
def mark_entities(tokens, tags):
    marked_tokens = []
    inside_entity = False
    for token, tag in zip(tokens, tags):
        if tag.startswith('B-'):
            if inside_entity:
                marked_tokens.append(']')
                inside_entity = False
            marked_tokens.append('[')
            marked_tokens.append(token)
            inside_entity = True
        elif tag.startswith('I-'):
            marked_tokens.append(token)
        else:
            if inside_entity:
                marked_tokens.append(']')
                inside_entity = False
            marked_tokens.append(token)
    if inside_entity:
        marked_tokens.append(']')
    return ''.join(marked_tokens)

def tokenize_text(text):
    # Split text into tokens, including brackets
    tokens = re.findall(r'\[|\]|[^\[\]\s]+', text)
    return tokens

def refined_alignment(source_tokens, source_tags, target_tokens):
    # Map positions of entities in source_tokens
    entity_positions = []
    idx = 0
    while idx < len(source_tags):
        tag = source_tags[idx]
        if tag.startswith('B-'):
            entity_type = tag[2:]
            start_idx = idx
            idx += 1
            while idx < len(source_tags) and source_tags[idx].startswith('I-'):
                idx += 1
            end_idx = idx - 1
            entity_positions.append((start_idx, end_idx, entity_type))
        else:
            idx += 1

    # Map positions of entities in target_tokens using brackets
    target_entity_positions = []
    idx = 0
    while idx < len(target_tokens):
        if target_tokens[idx] == '[':
            start_idx = idx + 1
            idx += 1
            while idx < len(target_tokens) and target_tokens[idx] != ']':
                idx += 1
            end_idx = idx - 1
            target_entity_positions.append((start_idx, end_idx))
            idx += 1  # Skip the closing ']'
        else:
            idx += 1

    # Check if the number of entities matches
    if len(entity_positions) != len(target_entity_positions):
        print("Mismatch between number of entities in source and target.")
        # You can choose to skip this sentence or handle it differently
        raise ValueError("Mismatch between number of entities in source and target.")

    # Assign tags to target tokens
    target_tags = ['O'] * len(target_tokens)
    for (start, end), (src_start, src_end, entity_type) in zip(target_entity_positions, entity_positions):
        if start >= len(target_tags) or end >= len(target_tags):
            print(f"Entity position out of range: start={start}, end={end}, len(target_tags)={len(target_tags)}")
            raise IndexError("Entity position out of range in target_tags.")
        target_tags[start] = 'B-' + entity_type
        for i in range(start + 1, end + 1):
            if i >= len(target_tags):
                print(f"Index {i} out of range for target_tags of length {len(target_tags)}")
                break
            target_tags[i] = 'I-' + entity_type

    # Remove brackets from tokens and tags
    tokens_without_brackets = []
    tags_without_brackets = []
    for token, tag in zip(target_tokens, target_tags):
        if token not in ['[', ']']:
            tokens_without_brackets.append(token)
            tags_without_brackets.append(tag)

    return tokens_without_brackets, tags_without_brackets

def levenshtein_distance(s1, s2):
    # s1 and s2 are lists of tokens
    size_x = len(s1) + 1
    size_y = len(s2) + 1
    matrix = [[0] * size_y for _ in range(size_x)]
    for x in range(size_x):
        matrix[x][0] = x  # Deletion
    for y in range(size_y):
        matrix[0][y] = y  # Insertion

    for x in range(1, size_x):
        for y in range(1, size_y):
            if s1[x - 1] == s2[y - 1]:
                substitution_cost = 0
            else:
                substitution_cost = 1
            matrix[x][y] = min(
                matrix[x - 1][y] + 1,      # Deletion
                matrix[x][y - 1] + 1,      # Insertion
                matrix[x - 1][y - 1] + substitution_cost  # Substitution
            )
    return matrix

def backtrace_operations(s1, s2, matrix):
    x, y = len(s1), len(s2)
    operations = []

    while x > 0 or y > 0:
        if x > 0 and y > 0 and matrix[x][y] == matrix[x - 1][y - 1] + (0 if s1[x - 1] == s2[y - 1] else 1):
            if s1[x - 1] == s2[y - 1]:
                operations.append(('match', s1[x - 1], s2[y - 1], x - 1, y - 1))
            else:
                operations.append(('sub', s1[x - 1], s2[y - 1], x - 1, y - 1))
            x -= 1
            y -= 1
        elif y > 0 and matrix[x][y] == matrix[x][y - 1] + 1:
            operations.append(('insert', '', s2[y - 1], x, y - 1))
            y -= 1
        else:
            operations.append(('delete', s1[x - 1], '', x - 1, y))
            x -= 1

    operations.reverse()
    return operations

# Process and translate a single sentence
def process_and_translate_sentence(sentence, translator):
    try:
        lines = sentence.strip().split("\n")
        sentences = [tuple(line.split()) for line in lines if line]
        source_tokens, source_tags = zip(*sentences)

        # Mark entities
        marked_text = mark_entities(source_tokens, source_tags)
        unmarked_text = ''.join(source_tokens)
        natural_text = ' '.join(source_tokens)  # Use spaces for natural translation

        # Translate the texts
        marked_translation = translator.call_url(marked_text)
        unmarked_translation = translator.call_url(unmarked_text)
        natural_translation = translator.call_url(natural_text)

        if marked_translation and unmarked_translation and natural_translation:
            # Tokenize the translations
            target_tokens_marked = tokenize_text(marked_translation.strip())
            target_tokens_unmarked = tokenize_text(unmarked_translation.strip())
            natural_tokens = tokenize_text(natural_translation.strip())

            # Use the marked translation to find entity positions
            tokens_without_brackets, target_tags = refined_alignment(
                source_tokens, source_tags, target_tokens_marked)

            # Compute alignment between code-generated translation and natural translation
            matrix = levenshtein_distance(tokens_without_brackets, natural_tokens)
            operations = backtrace_operations(tokens_without_brackets, natural_tokens, matrix)

            # Build the final tokens by replacing tokens where appropriate
            final_tokens = []
            final_tags = []
            idx_code = 0
            idx_natural = 0

            for op in operations:
                if op[0] == 'match':
                    # Tokens are the same
                    final_tokens.append(tokens_without_brackets[idx_code])
                    final_tags.append(target_tags[idx_code])
                    idx_code += 1
                    idx_natural += 1
                elif op[0] == 'sub':
                    # Tokens are different
                    if target_tags[idx_code] == 'O':
                        # Replace with natural translation
                        final_tokens.append(natural_tokens[idx_natural])
                    else:
                        # Keep the entity token
                        final_tokens.append(tokens_without_brackets[idx_code])
                    final_tags.append(target_tags[idx_code])
                    idx_code += 1
                    idx_natural += 1
                elif op[0] == 'insert':
                    # Insert token from natural translation
                    final_tokens.append(natural_tokens[idx_natural])
                    final_tags.append('O')
                    idx_natural += 1
                elif op[0] == 'delete':
                    # Delete token from code-generated translation
                    idx_code += 1

            # Output result
            result = '\n'.join([f"{word} {label}" for word, label in zip(final_tokens, final_tags)])
            return result
        else:
            print("Translation failed for one or more texts.")
        return None
    except Exception as e:
        print(f"Error processing sentence: {e}")
        return None

def main():
    input_path = 'input.txt'
    output_path = 'output.txt'
    translator = GetResult("ntrans.xfyun.cn")
    sentences = read_bio_data(input_path)
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for idx, sentence in enumerate(tqdm(sentences, desc="Processing", unit="sentence")):
            if sentence:
                try:
                    result = process_and_translate_sentence(sentence, translator)
                    if result:
                        outfile.write(result + '\n\n')
                except Exception as e:
                    print(f"Error processing sentence at index {idx}: {e}")
                    continue

if __name__ == "__main__":
    main()
