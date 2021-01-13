import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing as Winters

df = pd.read_csv('D:\\Docs\\Data Science\\ml-project2\\data\\sales_train.csv', parse_dates=['date'])
test_df = pd.read_csv('D:\\Docs\\Data Science\\ml-project2\\data\\test.csv', index_col = "ID")
df['item_cnt_day'] = df['item_cnt_day'].clip(0,20) # Вернём продажи в рамки, установленные в условии
# оставим в трейне только те магазины и итемы, которые просят предсказать
df = df[df['shop_id'].isin(test_df['shop_id'].unique())] 
df = df[df['item_id'].isin(test_df['item_id'].unique())]
# переведём в формат временных рядов для каждой пары итем-магазин
df = df.pivot_table(index=['shop_id', 'item_id'], values=['item_cnt_day'], columns=['date_block_num']
                     , fill_value=0, aggfunc='sum')
df = df.reset_index()

# Прекрасно, имеем временные ряды.

print(df["item_cnt_day"].iloc[0,])

# Сгладить, детрендировать и десезонизировать все эти ряды вручную чтобы воспользоваться AR\MA\ARMA я не могу, поэтому воспользуюсь моделью Винтерса

model = Winters(df["item_cnt_day"].loc[0,])
fitted = model.fit()
forecast = fitted.forecast(steps = 1)

# Посмотрев пример предсказания, понимаем, что мы малость завышаем - но всё нормально, округлим - работать будет

print(np.round(forecast))

# Всё, имеем фреймворк. Заloop'им получившийся алгоритм и готово!

train_pred = list()

data_for_pred = np.array(df['item_cnt_day'].values)

print(len(data_for_pred[0]))

# эта штука нужна для предсказания. запустил её только один раз, потом пользовался полученными предсказаниями, т.к. метод детерминированный (ну и долго считается, в районе 15 минут)

for i in range(len(data_for_pred)):
    model = Winters(data_for_pred[i])
    fitted = model.fit()
    forecast = np.round(fitted.forecast(steps = 1))
    train_pred.append(forecast)
    # Принтить шаг будем чисто чтобы понимать что что-то вообще работает
    print(i)

# Вся дальнейшая задача - грамотно присобачить предсказания и сделать сабмишн

predictions = pd.DataFrame(train_pred)

predictions = pd.read_csv("D:\\Docs\\Data Science\\ml-project2\\data\\predictions.csv", index_col = 0)

print(predictions)

# А здесь я записал получившийся вектор предсказаний (который соответствует трейну) в цсв чтобы не ждать по часу несколько раз

df = df.drop('item_cnt_day', axis = 1)
df['predictions'] = predictions['0'].clip(0, 20)

print(len(test_df))

# С чем я удачно справился

submission = pd.merge(test_df, df, how = "left", left_on = ["shop_id", "item_id"], right_on = ["shop_id", "item_id"])

submission.iloc[:,-1] = submission.iloc[:,-1].fillna(0.0)
submission = submission.drop("shop_id", axis = 1)
submission = submission.drop("item_id", axis = 1)
submission = submission.rename(columns={submission.columns[-1]:"item_cnt_month"})
submission["ID"] = submission.index
submission = submission[["ID", "item_cnt_month"]]
submission.to_csv("D:\\submission.csv", index=False)

# А ещё прошу прощения за .py файл - писал всегда на десктопе и ноутбуки кажутся мне не очень удобными