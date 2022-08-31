import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
tips= sns.load_dataset("tips") # importing the dataset maunally.
tips.describe ()
tips.head()
tips["day"] # inspect objects in the variable "Day" to rename the variable objects name
tips=tips.replace (['Thur', 'Fri', 'Sat', 'Sun'],['Thursday', 'Friday','Saturday','Sunday'])

fig= sns.relplot(data=tips, x="total_bill", y="tip", hue="day", col="sex")
fig.set(xlabel="Total bill in $",
        ylabel="Tibs in $",
       title="Relationship between tips and tips value")
fig.savefig("Python_Course_exercises/output/out_tips.pdf")

# Question number two

df = pd.read_csv("Python_Course_exercises/u.user.txt", sep='\t', na_values = "?")