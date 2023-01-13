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
turn opposite left = LTURN LTURN turn opposite right = RTURN RTURN
u opposite left = turn opposite left u
u opposite right = turn opposite right u
turn around left = LTURN LTURN LTURN LTURN
turn around right = RTURN RTURN RTURN RTURN
u around left = LTURN u LTURN u LTURN u LTURN u u around right = RTURN u RTURN u RTURN u RTURN u x twice = x x
x thrice = x x x
x1 and x2 = x1 x2
x1 after x2 = x2 x1
And these examples:
IN: run right twice and walk opposite right twice OUT: I_TURN_RIGHT I_RUN I_TURN_RIGHT I_RUN I_TURN_RIGHT I_TURN_RIGHT I_WALK I_TURN_RIGHT I_TURN_RIGHT I_WALK
IN: look right twice after turn right OUT: I_TURN_RIGHT I_TURN_RIGHT I_LOOK I_TURN_RIGHT I_LOOK
IN: look twice and run opposite left OUT: I_LOOK I_LOOK I_TURN_LEFT I_TURN_LEFT I_RUN
IN: walk around left thrice and walk right thrice OUT: I_TURN_LEFT I_WALK I_TURN_LEFT I_WALK I_TURN_LEFT I_WALK I_TURN_LEFT I_WALK I_TURN_LEFT I_WALK I_TURN_LEFT I_WALK I_TURN_LEFT I_WALK I_TURN_LEFT I_WALK I_TURN_LEFT I_WALK I_TURN_LEFT I_WALK I_TURN_LEFT I_WALK I_TURN_LEFT I_WALK I_TURN_RIGHT I_WALK I_TURN_RIGHT I_WALK I_TURN_RIGHT I_WALK
IN: run left after turn opposite right twice OUT: I_TURN_RIGHT I_TURN_RIGHT I_TURN_RIGHT I_TURN_RIGHT I_TURN_LEFT I_RUN
IN: look around left after turn opposite left twice OUT: I_TURN_LEFT I_TURN_LEFT I_TURN_LEFT I_TURN_LEFT I_TURN_LEFT I_LOOK I_TURN_LEFT I_LOOK I_TURN_LEFT I_LOOK I_TURN_LEFT I_LOOK
IN: turn left after run twice OUT: I_RUN I_RUN I_TURN_LEFT
IN: jump OUT: I_JUMP
Translate the following:
IN: turn left after jump twice OUT:"""
)
output = generator(prompt, max_new_tokens=50, max_length=None, num_return_sequences=10)

for i, sequence in enumerate(output):
    print(i, sequence['generated_text'][len(prompt):].split("\n")[0])
