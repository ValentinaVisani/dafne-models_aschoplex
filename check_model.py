import sys

import openai
from dafne_dl import DynamicDLModel

API_KEY_LOCATION = 'openai_api_key.txt'
api_key = open(API_KEY_LOCATION, 'r').read().strip()
openai.api_key = api_key

GPT_MODEL = 'gpt-3.5-turbo'
PROMPT = 'Analyze the following code and tell me if you think it contains malicious behavior, including attempt to access local files\n'

def main():
    model_file = sys.argv[1]
    # load the DynamicDLModel
    with open(model_file, 'rb') as f:
        model = DynamicDLModel.Load(f)

    # get the sources of the functions
    sources = {}
    for fn_name in model.function_mappings:
        try:
            src = getattr(model, fn_name).source
        except AttributeError:
            print('WARNING! Could not find source for', fn_name)
            continue
        sources[fn_name] = src

    # create the prompt
    for name, src in sources.items():
        prompt = PROMPT + src
        print('Analysis for ', name)
        response = openai.Completion.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": "You are a python senior developer that needs to do thorough code review to detect vulnerabilities."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
        )
        print(response)
        print()


if __name__ == '__main__':
    main()