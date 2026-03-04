#1 Read the csv file 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\satye\.spyder-py3\config\Practice\Costumer Segmentation in Betting Industry\project_dataset Costumer Segmentation Project (1).csv")

df.isnull().sum() #check the null values
df.duplicated().sum()
df["customer_id"].duplicated().sum()
df["customer_id"].nunique()
pd.set_option('display.max_columns', None)#for maximise the console 

df[df["customer_id"].duplicated()] #which rows are duplicated:
df["customer_id"][df["customer_id"].duplicated()].value_counts() #see the actual repeated IDs:
df[df["customer_id"].duplicated(keep=False)].sort_values("customer_id").head(10) #if the rows differ in other columns.


df_unique = df.groupby("customer_id",as_index=False).agg({
    "first_name": "first",
    "last_name": "first",
    "Gender": "first",
    "country": "first",
    "Total_Amount_Wagered": "mean",   # total batting amount
    "Age":"mean",
     "Salary":"mean" ,
     "Total_Number_of_Bets":"sum",
     "Number_of_Bonuses_Received":"sum",
     "Average_Amount_of_Bonuses_Received":"mean" 
}).reset_index()
df_unique.shape
print(df_unique["customer_id"].duplicated().sum())


df["customer_id"].value_counts().head(10)


df["customer_id"].value_counts().value_counts().plot(kind='bar')
plt.title("How Many Times Each Customer Appears")
plt.xlabel("Number of Occurrences")
plt.ylabel("Number of Customers")
plt.show()

# 

# 2.Salary Distribution
plt.figure(figsize=(8,5))
sns.histplot(df["Salary"], bins=30, edgecolor="black", color = "brown")
plt.title("Salary Distribution")
plt.xlabel("Salary")
plt.ylabel("Number of Customers")
plt.show()

# 3.Total Amount Wagered Distribution
plt.figure(figsize=(8,5))
sns.histplot(df["Total_Amount_Wagered"], bins=30, edgecolor="black", color="yellow")
plt.title("Total Amount Wagered Distribution")
plt.xlabel("Amount Wagered")
plt.ylabel("Number of Customers")
plt.show()

# 4.Bet Type Preferences (Bar Chart)
plt.figure(figsize=(8,5))
df["Bet_Type_Preference"].value_counts().plot(kind="bar", color="purple", edgecolor="black")
plt.title("Bet Type Preferences")
plt.xlabel("Bet Type")
plt.ylabel("Number of Customers")
plt.show()

# 5. Gender Distribution
plt.figure(figsize=(8,5))
df["Gender"].value_counts().plot(kind="pie")
plt.title("Bet Type Preferences")
plt.xlabel("Bet Type")
plt.ylabel("Number of Customers")
plt.show()

# 6. Age Distribution
plt.figure(figsize=(8,5))
sns.histplot(df["Age"], bins=20, edgecolor="black", color="green")
plt.title("Age Distribution of Customers")
plt.xlabel("Age")
plt.ylabel("Number of Customers")
plt.show()

# 7 Scatter Plot
df_numeric = df.select_dtypes(include=[np.number])
print("Shape after keeping only numeric columns:", df_numeric.shape)


plt.scatter(df_numeric["Salary"], df_numeric["Total_Amount_Wagered"])
plt.xlabel("Salary")
plt.ylabel("Total Amount Wagered")
plt.title("Scatter Plot of Features")
plt.show()

# 8. Correlation Heatmap (Salary, Wager, Bets)
plt.figure(figsize=(6,4))
sns.heatmap(df[["Salary","Total_Amount_Wagered","Total_Number_of_Bets"]].corr(), 
            annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()


# K- means Clustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

df = pd.read_csv(r"C:\Users\satye\Documents\project_data (1).csv")
print("The shape of the data is ", df.shape)
print(df.head())


#Elbow method /////////////////////////////////
wcss = []
for i in range(1, 11):
    km = KMeans(n_clusters=i, random_state=42)
    km.fit(df_numeric)   # fit only numeric data
    wcss.append(km.inertia_)

plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel("Number of clusters (k)")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()

kmeans = KMeans(n_clusters=3, random_state=42)
df["Cluster"] = kmeans.fit_predict(df_numeric) 


# Salary vs Total_Amount_Wagered
sns.scatterplot(data = df, x = "Salary", y = "Total_Amount_Wagered", hue = "Cluster", palette = "colorblind", s = 80, alpha= 1.0, edgecolor = "black")
plt.xlabel("Salary")
plt.ylabel("Total_Amount_Wagered")
plt.title("Clusters: Salary vs Total Amount Wagered")
plt.legend(title = "Cluster" )
plt.show()


# Total no. of bets vs No.of bonus received
plt.figure(figsize = (8, 6))
sns.scatterplot(x = "Total_Number_of_Bets", y = "Number_of_Bonuses_Received", data = df, hue = "Cluster", palette = "Set1", s = 80, alpha = 0.7, edgecolor = "black")
plt.title("Clusters :Total no. of bets vs No.of bonus received ")
plt.xlabel("Total_Number_of_Bets)")
plt.ylabel("Number_of_Bonuses_Received")
plt.legend(title = "Cluster")
plt.show()


# Age vs Total Number of Bets (sees if age effects betting frequency)
plt.figure(figsize = (8, 6))
sns.scatterplot(data = df, x = "Age", y = "Total_Number_of_Bets", hue = "Cluster",palette = "Set1", s = 80, alpha = 1.0, edgecolor = "black")
plt.title("Age vs Total_Number_of_Bets")
plt.xlabel("Age")
plt.ylabel("Total_Number_of_Bets")
plt.legend(title = "Cluster")
plt.show()


# Bonus Efficiency (shows how much bonus money a customer gets per bonus)
plt.figure(figsize = (8, 6))
sns.scatterplot(data = df, x = "Number_of_Bonuses_Received", y = "Average_Amount_of_Bonuses_Received", palette = "bright", hue = "Cluster", s = 80, alpha = 0.7, edgecolor = "black")
plt.title("Bonus Efficiency")
plt.xlabel("Number_of_Bonuses_Received")
plt.ylabel("Average_Amount_of_Bonuses_Received")
plt.legend(title = "Cluster")
plt.show()


# Total Amount wagered vs Total No. of Bets (Shows who bets more and who bets higher amounts)
plt.figure(figsize = (8, 6))
sns.scatterplot(data = df, x = "Total_Amount_Wagered", y = "Total_Number_of_Bets", palette = "muted", hue = "Cluster", alpha = 1.0, s = 80, edgecolor = "black")
plt.title("Total Amount wagered vs Total Number of Bets")
plt.xlabel("Total_Amount_Wagered")
plt.ylabel("Total_Number_of_Bets")
plt.legend(title = "Cluster")
plt.show()








