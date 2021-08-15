import pandas as pd

df = pd.read_csv("cleaned_news.csv")

X = df.drop(["fake_news"], axis = 1) 
Y = df["fake_news"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25)

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(max_features = 5000)
X_train_counts = count_vect.fit_transform(X_train["text"]) 
X_test = count_vect.transform(X_test["text"]) 

from sklearn.naive_bayes import MultinomialNB
Naive = MultinomialNB()
Naive.fit(X_train_counts, y_train)

from sklearn.metrics import accuracy_score
prediction = Naive.predict(X_test)
print(accuracy_score(predictions, y_test))

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = [‘auto’, ‘sqrt’]
max_depth = [int(x) for x in np.linspace(100, 500, num = 11)]
max_depth.append(None)
random_grid = {
 ‘n_estimators’: n_estimators,
 ‘max_features’: max_features,
 ‘max_depth’: max_depth
 }

 rfc_random = RandomizedSearchCV(estimator = rfc, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
 rfc_random.fit(X_train, y_train)
 print(rfc_random.best_params_)
 rfc = CountVectorizer(n_estimators=600, max_depth=300, max_features='sqrt')
rfc.fit(X_train,y_train)
rfc_predict = rfc.predict(X_test)
print(accuracy_score(rfc_predict, y_test)

final = 'US Senator Ben Sasse, a member of the Senate Select Committee on Intelligence, released a statement on Afghanistan Sunday, saying the US would "regret" its moves and foreign relations would suffer as a consequence.“The unmitigated disaster in Afghanistan – the shameful, Saigon-like abandonment of Kabul, the brutalization of Afghan women, and the slaughter of our allies – is the predictable outcome of the Trump-Biden doctrine of weakness," Sasse said in a blistering statement."History must be clear about this: American troops didn’t lose this war – Donald Trump and Joe Biden deliberately decided to lose. Politicians lied: America’s options were never simply this disgraceful withdrawal or an endless occupation force of 100,000 troops (we haven’t had that in Afghanistan in a decade)."Sasse continued that US leadership "did not tell the truth" over how crucial the nations peace-keeping force to Afghan security was.'
final_vec = count_vect.transform(final)
predict_final = Naive.predict(final_vec)
print(predict_final)
      
final = 'Having been informed by friends that such an item of clothing was essential in the Pacific Northwest, area man Walter Katrakis told reporters Friday he was shopping around for a nice fire-resistant jacket in anticipation of his move to Portland, OR. “I read that the Pacific Northwest can get up to 15 feet of fire a year, so I want to invest in something that will be sturdy enough to hold up through the [wildfire] season,” said Katrakis, 27, explaining that he was willing to shell out a bit more for a quality coat that wouldn’t immediately melt when conflagrations in the region hit 2,000 degrees Fahrenheit. “I also want it to be packable, so I can carry it with me and quickly throw it on if a nearby forest suddenly bursts into flames. Even though they’re pricey, I might get one of those vintage asbestos-woven firefighting suits. They look sharp, and they just don’t make them like that anymore.” At press time, Katrakis was trying to find something that matched the oxygen mask given to him by a friend who used to live in Oregon but left after his entire neighborhood burned to the ground.'
final_vec = count_vect.transform(final)
predict_final = Naive.predict(final_vec)
print(predict_final)
