from transformers import pipeline, set_seed

generator = pipeline('text-generation', model='gpt2-xl', device="cuda:0")
set_seed(0)

prompt = (
    """Given the grammar:
walk = WALK
look = LOOK
run = RUN
jump = JUMP
turn left = LTURN
turn right = RTURN
u left = LTURN u
u right = RTURN u
turn opposite left = LTURN LTURN
turn opposite right = RTURN RTURN
u opposite left = turn opposite left u
u opposite right = turn opposite right u
turn around left = LTURN LTURN LTURN LTURN
turn around right = RTURN RTURN RTURN RTURN
u around left = LTURN u LTURN u LTURN u LTURN u
u around right = RTURN u RTURN u RTURN u RTURN u
x twice = x x
x thrice = x x x
x1 and x2 = x1 x2
x1 after x2 = x2 x1
And these examples:
IN: run right twice and walk opposite right twice OUT: RTURN RUN RTURN RUN RTURN RTURN WALK RTURN RTURN WALK
IN: look right twice after turn right OUT: RTURN RTURN LOOK RTURN LOOK
IN: look twice and run opposite left OUT: LOOK LOOK LTURN LTURN RUN
IN: walk around left thrice and walk right thrice OUT: LTURN WALK LTURN WALK LTURN WALK LTURN WALK LTURN WALK LTURN WALK LTURN WALK LTURN WALK LTURN WALK LTURN WALK LTURN WALK LTURN WALK RTURN WALK RTURN WALK RTURN WALK
IN: run left after turn opposite right twice OUT: RTURN RTURN RTURN RTURN LTURN RUN
IN: look around left after turn opposite left twice OUT: LTURN LTURN LTURN LTURN LTURN LOOK LTURN LOOK LTURN LOOK LTURN LOOK
IN: turn left after run twice OUT: RUN RUN LTURN
IN: jump OUT: JUMP
Translate the following:
IN: turn left after jump twice OUT:"""
)
output = generator(prompt, max_new_tokens=50,
                   max_length=None, num_return_sequences=10)

for i, sequence in enumerate(output):
    print(i, sequence['generated_text'][len(prompt):].split("\n")[0])
