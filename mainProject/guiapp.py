import tkinter as tk
from joblib import dump, load
import pandas as pd

def buildWidget(master,text,x,y):
    widget_l = tk.Label(
        master,
        text=text,
        font=('Calibri', 14)
    )
    widget_l.place(relx=x, rely=y, anchor="center")

    widget_e = tk.Entry(
        master,
        width=18
    )

    widget_e.place(relx=x+0.215, rely=y,anchor="center")

    return widget_l, widget_e


def main():
    keylist = ['global_sentiment_polarity', 'num_hrefs', 'n_non_stop_unique_tokens',
               'kw_avg_min', 'kw_min_avg', 'global_rate_positive_words',
               'n_unique_tokens', 'global_subjectivity', 'kw_max_min',
               'n_tokens_content', 'kw_max_avg']

    textlist_c = ["Number of links:",
                "Worst keyword (avg. shares)",
                "Avg. keyword (min. shares)",
                "Worst keyword (max. shares)",
                "Avg. keyword (max. shares)",
                "Number of words in the content",]

    textlist_r = ["Text sentiment polarity(0-1)",
                "Rate of unique non-stop words in the content(0-1)",
                "Rate of positive words in the content(0-1)",
                "Text subjectivity(0-1)",
                "Rate of unique words in the content(0-1)"]

    order_list = [
        'num_hrefs',
        'kw_avg_min',
        'kw_min_avg',
        'kw_max_min',
        'kw_max_avg',
        'n_tokens_content',
        'global_sentiment_polarity',
        'n_non_stop_unique_tokens',
        'global_rate_positive_words',
        'global_subjectivity',
        'n_unique_tokens']



    class App(tk.Frame):
        def __init__(self, master):
            self.creatWidget(master)

        def creatWidget(self, master):
            self.tittle = tk.Label(
                master,
                text='Prediction of the popularity of business news articles',
                font=('Calibri', 22, 'bold')
            )
            self.tittle.place(relx=0.25, rely=0.01)


            self.subtittle1 = tk.Label(
                master,
                text="Input of constant feature values",
                font=('Calibri', 16, 'bold')
            )
            self.subtittle1.place(relx=0.4, rely=0.075)

            xc = 0.425
            yc_start = 0.15

            for i in textlist_c:
                self.cl, self.ce = buildWidget(master, i, x=xc, y=yc_start)
                yc_start += 0.05

            self.subtittle2 = tk.Label(
                master,
                text="Input of ratio features",
                font=('Calibri', 16, 'bold')
            )
            self.subtittle2.place(relx=0.425, rely=0.425)

            xr = 0.425
            yr_start = 0.5

            for i in textlist_r:
                self.rl, self.re = buildWidget(master, i, x=xr, y=yr_start)
                yr_start += 0.05


            self.button = tk.Button(
                master,
                text="Calculate",
                font=('Calibri', 16, 'bold')
            )
            self.button.place(relx=0.5, rely=0.725)

            def click(event):
                res_dic = {}
                cal_dic = {}
                children = master.winfo_children()
                res_list_click = []
                for i in children:
                    child_type = str(type(i))
                    if child_type == "<class 'tkinter.Entry'>":
                        res_list_click.append(float(i.get().replace(' ', '')))

                for a, b in enumerate(res_list_click):
                    key = order_list[a]
                    val = b
                    res_dic.update({key: val})

                for key in keylist:
                    k = key
                    v = [res_dic.get(key)]
                    cal_dic.update({k:v})

                # print(res_list_click)
                # print(res_dic)
                print(cal_dic)

                rf = load('rfcFinal.joblib')
                df = pd.DataFrame(cal_dic)
                pro = rf.predict_proba(df)[0, 1]
                res = round(pro * 100)
                print(res)

                self.res = tk.Label(
                    master,
                    text='The likelihood of this news article being popular is {}%'.format(res),
                    font=('Calibri', 20, 'bold')
                )
                self.res.place(relx=0.3, rely=0.825)

            self.button.bind("<Button-1>", click)





    main_window = tk.Tk()
    main_window.geometry('1200x800+400+100')
    main_window.title("Prediction of the popularity")

    application = App(main_window)
    main_window.mainloop()

if __name__ == '__main__':
    main()