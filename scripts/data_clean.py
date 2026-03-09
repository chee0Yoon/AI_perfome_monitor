# %%
import pandas as pd
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
correct_data = pd.read_csv(DATA_DIR / "correct.csv")
incorrect_data = pd.read_csv(DATA_DIR / "incorrect.csv")


# %%
def split_system_user(str):
    data = json.loads(str)
    print(data)
    for row in data:
        try:
            role = row.get('role', None)
            content = row.get('content', None)
            
            if role == 'system':
                system = content
            elif role == 'user':
                user = content
        except:
            system = ""
            user = data    
        
    return system, user


# %%
correct_data['Prompt'], correct_data['input'] = zip(*correct_data['input'].apply(split_system_user))
incorrect_data['Prompt'], incorrect_data['input'] = zip(*incorrect_data['input'].apply(split_system_user))


# %%
correct_data = correct_data[['id','Prompt', 'input','expectedOutput']]
correct_data['eval'] = 'correct'
incorrect_data = incorrect_data[['id','Prompt', 'input','expectedOutput']]
incorrect_data['eval'] = 'incorrect'
# %%
ilearning_data = pd.concat([correct_data, incorrect_data], ignore_index=True)


def safe_parse_output(x):
    try:
        return json.loads(x)['content']
    except:
        return x

ilearning_data['expectedOutput']= ilearning_data['expectedOutput'].apply(safe_parse_output)

# %%

ilearning_data.to_csv(DATA_DIR / "ilearning_data.csv", index=False)
# %%

humminggo_data = pd.DataFrame()
sheet_names = ["김과장님", "송부장님"]
for sheet in sheet_names:
    temp = pd.read_excel(DATA_DIR / "[HUMMINGo] Human Annotation Result.xlsx", sheet_name=sheet)
    humminggo_data = pd.concat([humminggo_data, temp], ignore_index=True)
    

humminggo_data = humminggo_data.rename(columns={'idx':'id', 'system_prompt':'Prompt', 'user_input':'input', 'assistant_output':'expectedOutput'})
humminggo_data['input'] = humminggo_data['input'].fillna(humminggo_data['user_prompt'])
humminggo_data['eval'] = humminggo_data['human_feedback_category'].apply(lambda x: 'correct' if x in ['Acceptable','Idea'] else 'incorrect')
# %%
humminggo_data.to_csv(DATA_DIR / "humminggo_data.csv", index=False)
# %%
