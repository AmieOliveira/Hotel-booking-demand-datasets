"""
    Desafio Hurb
"""

import pandas as pd
datapath = "hotel_bookings.csv"

from sklearn.preprocessing import LabelEncoder
from sklearn import tree


def tree_classification_withSolverColumn():
    # Pre-processing
    input_data = reservas.copy()
    labels = input_data.pop("is_canceled")
    # res.drop("reservation_status", inplace=True, axis=1)          # TODO! This column should have the result
    # res.drop("reservation_status_date", inplace=True, axis=1)

    # months = {"January":1, "February":2, "March":3, "April":4, "May":5, "June":6,
    #          "July":7, "August":8, "September":9, "October":10, "November":11, "December":12}

    encoders = {}

    for col in input_data.columns:
        if input_data[col].dtype == 'object':
            print(col)
            enc = LabelEncoder()

            # if col == "arrival_date_month":
            #    enc.set_params(**months)
            # else:

            try:
                enc.fit(input_data[col])
            except TypeError:
                input_data[col].fillna("No Data", inplace=True)
                enc.fit(input_data[col])

            input_data[col] = enc.transform(input_data[col])
            encoders[col] = enc

        elif input_data[col].isnull().values.any():
            input_data[col].fillna(-1, inplace=True)


    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(input_data, labels)

    # Analysis
    feature_name = list(input_data.columns)
    class_name = ["Ok", "Canceled"]
    tree.plot_tree(clf)
    r = tree.export_text(clf, feature_names=feature_name)
    print(r)

    ## Graphviz visualization
    #import graphviz
    #dot_data = tree.export_graphviz(resulting_trees[0], out_file=None,
    #                                feature_names=feature_name,
    #                                class_names=class_name,
    #                                filled=True, rounded=True,
    #                                special_characters=True)
    #graph = graphviz.Source(dot_data)
    #graph.render("reservas_arvore-1")


if __name__ == '__main__':
    reservas = pd.read_csv("hotel_bookings.csv")
    print(reservas)
    print(reservas.dtypes)
    print(reservas.info())

    # As plots não ficaram como eu queria...

    # print(reservas[["customer_type", "company"]].head())
    counts = reservas["customer_type"].value_counts()
    print(counts)
    counts.plot.bar()

    # reservas.boxplot(by="is_canceled", figsize=(15,20))

    # df = reservas[['customer_type','required_car_parking_spaces']]
    #
    # enc = LabelEncoder()
    # enc.fit(df['customer_type'])
    # print(enc.classes_)
    # print(enc.get_params(deep=True))
    # df['customer_type'] = enc.transform(df['customer_type'])
    #
    # print(df.head(10))
    #
    # df.plot.scatter(x="customer_type", y="required_car_parking_spaces", alpha=0.5)

    # reservas[['is_canceled', 'hotel']].plot.hist(by='hotel')

    # fig, ax = plt.subplots(figsize=(8, 6))
    # reservas[['is_canceled', 'hotel', 'reservation_status_date']].
    # groupby(['reservation_status_date', 'hotel']).plot(kind='hist', ax=ax)

    tree_classification_withSolverColumn()