"""Writes the GSM8K test set in the format of data.data_path (configs/data/gsm8k-test.yaml):
a JSON list of {"prompt": question, "response_ground_truth": answer}, where the answer is
the GSM8K solution ending with '#### <number>' (read by sandbox_gsm8k._extract_gold_answer).

Usage (from the repo root): python scripts/prepare_gsm8k_test.py [output_path]
"""

import json
import os
import sys

import datasets


def main():
  out_path = sys.argv[1] if len(sys.argv) > 1 else 'data/gsm8k_test.json'
  ds = datasets.load_dataset('openai/gsm8k', 'main', split='test')
  records = [{'prompt': ex['question'],
              'response_ground_truth': ex['answer']} for ex in ds]
  os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
  with open(out_path, 'w') as f:
    json.dump(records, f, indent=2)
  print(f'Wrote {len(records)} GSM8K test examples to {out_path}')


if __name__ == '__main__':
  main()
