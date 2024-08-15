import logging
import onnxruntime_genai as og
from flask import Flask, request, jsonify

app = Flask(__name__)

class Args:
    def __init__(self) -> None:
        self.model = "cuda/cuda-int4-rtn-block-32"
        self.do_sample = True
        self.max_length = 4096
        self.min_length = None
        self.top_p = None
        self.top_k = None
        self.temperature = 0.00
        self.repetition_penalty = None


class Phi:
    def __init__(self) -> None:
        pass

    def generate_phi_4k(self, text) -> dict:
        args = Args()
        model = og.Model(f'{args.model}')
        tokenizer = og.Tokenizer(model)
        tokenizer_stream = tokenizer.create_stream()
        search_options = {name: getattr(args, name) for name in ['do_sample', 'max_length', 'min_length', 'top_p', 'top_k', 'temperature', 'repetition_penalty'] if hasattr(args, name) and getattr(args, name) is not None}
        if 'max_length' not in search_options:
            search_options['max_length'] = 2048

        chat_template = '<|user|>\n{input}<|end|>\n<|assistant|>'
        prompt = f'{chat_template.format(input=text)}'
        input_tokens = tokenizer.encode(prompt)

        params = og.GeneratorParams(model)
        params.set_search_options(**search_options)
        params.input_ids = input_tokens
        generator = og.Generator(model, params)

        print()
        print("Output: ", end='', flush=True)
        response =[]
        try:
            while not generator.is_done():
                generator.compute_logits()
                generator.generate_next_token()

                new_token = generator.get_next_tokens()[0]
                print(tokenizer_stream.decode(new_token), end='', flush=True)
                response.append(tokenizer_stream.decode(new_token))
        except KeyboardInterrupt:
            print("  --control+c pressed, aborting generation--")
        print()
        print()
        del generator
        return ''.join(response)

@app.route('/generate', methods=['POST'])
def generate_text() -> dict:
    if not request.json or 'text' not in request.json:
        return jsonify({'error': 'Missing text'}), 400

    text = request.json['text']
    phi_model = Phi()
    generated_text = phi_model.generate_phi_4k(text)

    return jsonify({'generated_text': generated_text})




if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5201)

# test endpoint
# curl -X POST http://localhost:5201/generate -H "Content-Type: application/json" -d "{\"text\": \"How are transfomrers trained?\"}"
