import pandas as pd

fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")
true["fake_news"] = 0 
fake["fake_news"] = 1

just_text = true["text"]
just_text = just_text.str.extractall(r"^.*? - (?P<text>.*)")
just_text = just_text.droplevel(1)
true = true.assign(text=just_text["text"])

df = pd.concat([fake, true], axis = 0)
df = df.drop(["subject", "date", "title"], axis = 1) 
clean_text = df.to_csv("cleaned_news.csv", index = False)

