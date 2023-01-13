from transformers import pipeline, set_seed

generator = pipeline(
    'text2text-generation', model='google/flan-t5-xxl', device="cuda:0")
set_seed(0)

prompt = (
    """Given these examples:
IN: run right twice and walk opposite right twice OUT: RTURN RUN RTURN RUN RTURN RTURN WALK RTURN RTURN WALK
IN: look right twice after turn right OUT: RTURN RTURN LOOK RTURN LOOK
IN: look twice and run opposite left OUT: LOOK LOOK LTURN LTURN RUN
IN: walk around left thrice and walk right thrice OUT: LTURN WALK LTURN WALK LTURN WALK LTURN WALK LTURN WALK LTURN WALK LTURN WALK LTURN WALK LTURN WALK LTURN WALK LTURN WALK LTURN WALK RTURN WALK RTURN WALK RTURN WALK
IN: run left after turn opposite right twice OUT: RTURN RTURN RTURN RTURN LTURN RUN
IN: look around left after turn opposite left twice OUT: LTURN LTURN LTURN LTURN LTURN LOOK LTURN LOOK LTURN LOOK LTURN LOOK
IN: turn left after run twice OUT: RUN RUN LTURN
IN: jump OUT: JUMP
Translate the sentence:
turn left after jump twice"""
)

output = generator(prompt, max_new_tokens=50,
                   max_length=None)

print(output)

for i, sequence in enumerate(output):
    print(f'{i}:', sequence['generated_text'][len(prompt):].split("\n")[0])
