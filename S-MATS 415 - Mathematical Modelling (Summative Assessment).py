#!/usr/bin/env python
# coding: utf-8

# ![HEADER-5.png](attachment:HEADER-5.png)

# In[236]:


# IMPORTING PYTHON LIBRARIES
import nltk
import warnings
import wordcloud
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from nrclex import NRCLex
from matplotlib import pyplot
from tqdm.notebook import tqdm
from collections import Counter
from pmdarima import auto_arima
from wordcloud import WordCloud, STOPWORDS
from statsmodels.tsa.arima.model import ARIMA
from matplotlib.collections import PolyCollection
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from sklearn.model_selection import train_test_split
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.metrics import mean_absolute_percentage_error
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# BABY PACIFIER DATA FRAME
BabyPacifier = pd.read_csv("BabyPacifier.csv", encoding = "latin-1")
BabyPacifier.index = BabyPacifier.index + 1
BabyPacifier.head()


# In[3]:


# HAIR DRYER DATA FRAME
HairDryer = pd.read_csv("HairDryer.csv", encoding = "latin-1")
HairDryer.index = HairDryer.index + 1
HairDryer.head()


# In[4]:


# MICROWAVE OVEN DATA FRAME
MicrowaveOven = pd.read_csv("MicrowaveOven.csv", encoding = "latin-1")
MicrowaveOven.index = MicrowaveOven.index + 1
MicrowaveOven.head()


# In[5]:


# COMBINATING ALL DATA SETS
AllDataFrame = [BabyPacifier, HairDryer, MicrowaveOven]
DataSet = pd.concat(AllDataFrame)
display(DataSet)


# In[6]:


# PRODUCT CATEGORY DESCRIPTIVES
ProductCategoryDescriptives = DataSet.groupby("PRODUCT CATEGORY").size().to_frame("TOTAL")
ProductCategoryDescriptivesDataFrame = pd.DataFrame(ProductCategoryDescriptives)
ProductCategoryDescriptivesDataFrame.T


# In[7]:


# BABY PACIFIER STAR RATINGS
BabyPacifierDescriptives = DataSet[DataSet["PRODUCT CATEGORY"] == "Baby"]
BabyDescriptives = BabyPacifierDescriptives.groupby(["PRODUCT CATEGORY", "STAR RATING"]).size().to_frame("COUNT")
BabyDataFrame = pd.DataFrame(BabyDescriptives)
BabyDataFrame.T


# In[8]:


# HAIR DRYER STAR RATINGS
HairDryerDescriptives = DataSet[DataSet["PRODUCT CATEGORY"] == "Beauty"]
BeautyDescriptives = HairDryerDescriptives.groupby(["PRODUCT CATEGORY", "STAR RATING"]).size().to_frame("COUNT")
BeautyDataFrame = pd.DataFrame(BeautyDescriptives)
BeautyDataFrame.T


# In[9]:


# MICROWAVE OVEN STAR RATINGS
MicrowaveOvenDescriptives = DataSet[DataSet["PRODUCT CATEGORY"] == "Major Appliances"]
MajorAppliancesDescriptives = MicrowaveOvenDescriptives.groupby(["PRODUCT CATEGORY", "STAR RATING"]).size().to_frame("COUNT")
MajorAppliancesDataFrame = pd.DataFrame(MajorAppliancesDescriptives)
MajorAppliancesDataFrame.T


# In[10]:


# COLOR SETS
OneStar = ["#BEABCF", "#73D3AA", "#F2D9A0"]
TwoStar = ["#A389BB", "#49C791", "#ECC670"]
ThreeStar = ["#8967A7", "#34A675", "#E5B03F"]
FourStar = ["#6D4F89", "#277D58", "#D39B1D"]
FiveStar = ["#412F51", "#133D2B", "#876313"]


# In[11]:


# BABY PACIFIER COLOR PATCHES
BP5 = mpatches.Patch(color = "#412F51", label = "★★★★★ : %s" % BabyDataFrame.COUNT[4])
BP4 = mpatches.Patch(color = "#6D4F89", label = "★★★★☆ : %s" % BabyDataFrame.COUNT[3])
BP3 = mpatches.Patch(color = "#8967A7", label = "★★★☆☆ : %s" % BabyDataFrame.COUNT[2])
BP2 = mpatches.Patch(color = "#A389BB", label = "★★☆☆☆ : %s" % BabyDataFrame.COUNT[1])
BP1 = mpatches.Patch(color = "#BEABCF", label = "★☆☆☆☆ : %s" % BabyDataFrame.COUNT[0])
BP0 = mpatches.Patch(color = "#FFFFFF")


# In[12]:


# HAIR DRYER COLOR PATCHES
HD5 = mpatches.Patch(color = "#133D2B", label = "★★★★★ : %s" % BeautyDataFrame.COUNT[4])
HD4 = mpatches.Patch(color = "#277D58", label = "★★★★☆ : %s" % BeautyDataFrame.COUNT[3])
HD3 = mpatches.Patch(color = "#34A675", label = "★★★☆☆ : %s" % BeautyDataFrame.COUNT[2])
HD2 = mpatches.Patch(color = "#49C791", label = "★★☆☆☆ : %s" % BeautyDataFrame.COUNT[1])
HD1 = mpatches.Patch(color = "#73D3AA", label = "★☆☆☆☆ : %s" % BeautyDataFrame.COUNT[0])


# In[13]:


# MICROWAVE OVEN COLOR PATCHES
MO5 = mpatches.Patch(color = "#876313", label = "★★★★★ : %s" % MajorAppliancesDataFrame.COUNT[4])
MO4 = mpatches.Patch(color = "#D39B1D", label = "★★★★☆ : %s" % MajorAppliancesDataFrame.COUNT[3])
MO3 = mpatches.Patch(color = "#E5B03F", label = "★★★☆☆ : %s" % MajorAppliancesDataFrame.COUNT[2])
MO2 = mpatches.Patch(color = "#ECC670", label = "★★☆☆☆ : %s" % MajorAppliancesDataFrame.COUNT[1])
MO1 = mpatches.Patch(color = "#F2D9A0", label = "★☆☆☆☆ : %s" % MajorAppliancesDataFrame.COUNT[0])


# In[14]:


# BAR PLOT CALCULATIONS
x = sorted(list(DataSet["PRODUCT CATEGORY"].unique()))
y1 = np.array([BabyDataFrame.COUNT[0], BeautyDataFrame.COUNT[0], MajorAppliancesDataFrame.COUNT[0]])
y2 = np.array([BabyDataFrame.COUNT[1], BeautyDataFrame.COUNT[1], MajorAppliancesDataFrame.COUNT[1]])
y3 = np.array([BabyDataFrame.COUNT[2], BeautyDataFrame.COUNT[2], MajorAppliancesDataFrame.COUNT[2]])
y4 = np.array([BabyDataFrame.COUNT[3], BeautyDataFrame.COUNT[3], MajorAppliancesDataFrame.COUNT[3]])
y5 = np.array([BabyDataFrame.COUNT[4], BeautyDataFrame.COUNT[4], MajorAppliancesDataFrame.COUNT[4]])
ArrayTotal = y1 + y2 + y3 + y4 + y5
y1 = (y1/ArrayTotal*100)
y2 = (y2/ArrayTotal*100)
y3 = (y3/ArrayTotal*100)
y4 = (y4/ArrayTotal*100)
y5 = (y5/ArrayTotal*100)


# In[15]:


# STAR RATING STACKED BAR PLOT VISUALIZATION
plt.figure(figsize = (10, 7))
plt.grid(color = "black", linestyle = "--", linewidth = 1, axis = "y", alpha = 0.3)
Grid = plt.gca()
Grid.set_axisbelow(True)
plt.bar(x, y1, color = OneStar)
plt.bar(x, y2, bottom = y1, color = TwoStar)
plt.bar(x, y3, bottom = y1 + y2, color = ThreeStar)
plt.bar(x, y4, bottom = y1 + y2 + y3, color = FourStar)
plt.bar(x, y5, bottom = y1 + y2 + y3 + y4, color = FiveStar)
for xpos, ypos, yval in zip(x, y1/2, y1):
    plt.text(xpos, ypos, "%.2f" % yval + "%", ha = "center", va = "center")
for xpos, ypos, yval in zip(x, y1 + y2/2, y2):
    plt.text(xpos, ypos, "%.2f" % yval + "%", ha = "center", va = "center")
for xpos, ypos, yval in zip(x, y1 + y2 + y3/2, y3):
    plt.text(xpos, ypos, "%.2f" % yval + "%", ha = "center", va = "center")
for xpos, ypos, yval in zip(x, y1 + y2 + y3 + y4/2, y4):
    plt.text(xpos, ypos, "%.2f" % yval + "%", ha = "center", va = "center")
for xpos, ypos, yval in zip(x, y1 + y2 + y3 + y4 + y5/2, y5):
    plt.text(xpos, ypos, "%.2f" % yval + "%", ha = "center", va = "center")
for xpos, ypos, yval in zip(x, y1 + y2 + y3 + y4 + y5, ArrayTotal):
    plt.text(xpos, ypos, "TOTAL = %d\n" % yval, ha = "center", va = "bottom")
Labels = ["Baby Pacifier", "Hair Dryer", "Microwave Oven"]
plt.xlabel("\nSUNSHINE COMPANY PRODUCTS")
plt.ylabel("PERCENTAGE BY STAR RATINGS\n")
plt.ylim(0, 110)
plt.yticks(np.arange(0, 110, 10))
plt.xticks(x, Labels)
plt.legend(handles = [BP5, BP4, BP3, BP2, BP1, BP0, HD5, HD4, HD3, HD2, HD1, BP0, MO5, MO4, MO3, MO2, MO1],
           bbox_to_anchor = (1.01, 0.5), title = "STAR RATING COUNTS:\n", loc = "center left")


# In[16]:


# BABY PACIFIER REVIEWS WORDCLOUD VISUALIZATION
BPCommentWords = ""
BPStopwords = list(STOPWORDS) + ["will"]
BabyPacifier.rename(columns = {"REVIEW BODY": "REVIEWBODY"}, inplace = True)
for val in BabyPacifier.REVIEWBODY:
    val = str(val)
    tokens = val.split()
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()
    BPCommentWords += " ".join(tokens) + " " 
BPWordCloud = WordCloud(width = 800, height = 600, background_color = "white", stopwords = BPStopwords, 
                        min_font_size = 10, max_words = 50, mask = None, colormap = "Purples",
                        random_state = 0, collocations = False).generate(BPCommentWords)                    
BabyPacifier.rename(columns = {"REVIEWBODY": "REVIEW BODY"}, inplace = True)
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(BPWordCloud)
plt.axis("off")
plt.title("BABY PACIFIER\n", fontdict = {"fontsize": 20})
plt.tight_layout(pad = 0)
plt.show()


# In[17]:


# HAIR DRYER REVIEWS WORDCLOUD VISUALIZATION
HDCommentWords = ""
HDStopwords = list(STOPWORDS) + ["will", "br"]
HairDryer.rename(columns = {"REVIEW BODY": "REVIEWBODY"}, inplace = True)
for val in HairDryer.REVIEWBODY:
    val = str(val)
    tokens = val.split()
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()
    HDCommentWords += " ".join(tokens) + " " 
HDWordCloud = WordCloud(width = 800, height = 600, background_color = "white", stopwords = HDStopwords, 
                        min_font_size = 10, max_words = 50, mask = None, colormap = "Greens",
                        random_state = 4, collocations = False).generate(HDCommentWords)                    
HairDryer.rename(columns = {"REVIEWBODY": "REVIEW BODY"}, inplace = True)
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(HDWordCloud)
plt.axis("off")
plt.title("HAIR DRYER\n", fontdict = {"fontsize": 20})
plt.tight_layout(pad = 0)
plt.show()


# In[18]:


# MICROWAVE OVEN REVIEWS WORDCLOUD VISUALIZATION
MOCommentWords = ""
MOStopwords = list(STOPWORDS) + ["will", "ge"]
MicrowaveOven.rename(columns = {"REVIEW BODY": "REVIEWBODY"}, inplace = True)
for val in MicrowaveOven.REVIEWBODY:
    val = str(val)
    tokens = val.split()
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()
    MOCommentWords += " ".join(tokens) + " " 
MOWordCloud = WordCloud(width = 800, height = 600, background_color = "white", stopwords = MOStopwords, 
                        min_font_size = 10, max_words = 50, mask = None, colormap = "Reds",
                        random_state = 3, collocations = False).generate(MOCommentWords)                    
MicrowaveOven.rename(columns = {"REVIEWBODY": "REVIEW BODY"}, inplace = True)
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(MOWordCloud)
plt.axis("off")
plt.title("MICROWAVE OVEN\n", fontdict = {"fontsize": 20})
plt.tight_layout(pad = 0)
plt.show()


# In[19]:


# 1-STAR RATING WORD SET
WordSet1Star = DataSet[DataSet['STAR RATING'] == 1]['REVIEW BODY']
WordSetRating1 = pd.DataFrame(WordSet1Star)
WordSetCounter1 = Counter(" ".join(WordSetRating1["REVIEW BODY"].str.lower()).split()).most_common(900)
WordSetDataFrame1 = pd.DataFrame(WordSetCounter1, columns = ["WORDS", "COUNT"])
WordSetDataFrame1.index = WordSetDataFrame1.index + 1
WordSetDataFrame1


# In[20]:


# 2-STAR RATING WORD SET
WordSet2Star = DataSet[DataSet['STAR RATING'] == 2]['REVIEW BODY']
WordSetRating2 = pd.DataFrame(WordSet2Star)
WordSetCounter2 = Counter(" ".join(WordSetRating2["REVIEW BODY"].str.lower()).split()).most_common(900)
WordSetDataFrame2 = pd.DataFrame(WordSetCounter2, columns = ["WORDS", "COUNT"])
WordSetDataFrame2.index = WordSetDataFrame2.index + 1
WordSetDataFrame2


# In[21]:


# 3-STAR RATING WORD SET
WordSet3Star = DataSet[DataSet['STAR RATING'] == 3]['REVIEW BODY']
WordSetRating3 = pd.DataFrame(WordSet3Star)
WordSetCounter3 = Counter(" ".join(WordSetRating3["REVIEW BODY"].str.lower()).split()).most_common(900)
WordSetDataFrame3 = pd.DataFrame(WordSetCounter3, columns = ["WORDS", "COUNT"])
WordSetDataFrame3.index = WordSetDataFrame3.index + 1
WordSetDataFrame3


# In[22]:


# 4-STAR RATING WORD SET
WordSet4Star = DataSet[DataSet['STAR RATING'] == 4]['REVIEW BODY']
WordSetRating4 = pd.DataFrame(WordSet4Star)
WordSetCounter4 = Counter(" ".join(WordSetRating4["REVIEW BODY"].str.lower()).split()).most_common(900)
WordSetDataFrame4 = pd.DataFrame(WordSetCounter4, columns = ["WORDS", "COUNT"])
WordSetDataFrame4.index = WordSetDataFrame4.index + 1
WordSetDataFrame4


# In[23]:


# 5-STAR RATING WORD SET
WordSet5Star = DataSet[DataSet['STAR RATING'] == 5]['REVIEW BODY']
WordSetRating5 = pd.DataFrame(WordSet5Star)
WordSetCounter5 = Counter(" ".join(WordSetRating5["REVIEW BODY"].str.lower()).split()).most_common(900)
WordSetDataFrame5 = pd.DataFrame(WordSetCounter5, columns = ["WORDS", "COUNT"])
WordSetDataFrame5.index = WordSetDataFrame5.index + 1
WordSetDataFrame5


# In[24]:


# WORD SET INTERSECTIONS
Set1 = set(WordSetDataFrame1["WORDS"].values.tolist())
Set2 = set(WordSetDataFrame2["WORDS"].values.tolist())
Set3 = set(WordSetDataFrame3["WORDS"].values.tolist())
Set4 = set(WordSetDataFrame4["WORDS"].values.tolist())
Set5 = set(WordSetDataFrame5["WORDS"].values.tolist())
SetIntersection = list(Set1.intersection(Set2, Set3, Set4, Set5))
print(SetIntersection)


# In[25]:


# WORD SET SUMMARY
warnings.filterwarnings("ignore")
Words = "enough|easy|cheap|almost|great|difficult|different|extreme|least|expensive|happy|exactly|best|cute|suck"
WordSet1 = WordSetDataFrame1[WordSetDataFrame1["WORDS"].str.contains(Words)].sort_values("WORDS", ascending = True)
WordSet2 = WordSetDataFrame2[WordSetDataFrame2["WORDS"].str.contains(Words)].sort_values("WORDS", ascending = True)
WordSet3 = WordSetDataFrame3[WordSetDataFrame3["WORDS"].str.contains(Words)].sort_values("WORDS", ascending = True)
WordSet4 = WordSetDataFrame4[WordSetDataFrame4["WORDS"].str.contains(Words)].sort_values("WORDS", ascending = True)
WordSet5 = WordSetDataFrame5[WordSetDataFrame5["WORDS"].str.contains(Words)].sort_values("WORDS", ascending = True)
Merge1 = pd.merge(WordSet1, WordSet2[['WORDS','COUNT']], on = 'WORDS', how = 'left')
Merge2 = pd.merge(Merge1, WordSet3[['WORDS','COUNT']], on = 'WORDS', how = 'left')
Merge3 = pd.merge(Merge2, WordSet4[['WORDS','COUNT']], on = 'WORDS', how = 'left')
Merge4 = pd.merge(Merge3, WordSet5[['WORDS','COUNT']], on = 'WORDS', how = 'left')
WordSetSummary = Merge4.set_axis(["WORDS", "1-STAR", "2-STAR", "3-STAR", "4-STAR", "5-STAR"], axis = 1, inplace = False)
WordSetSummary["FREQUENCY"] = WordSetSummary[["1-STAR", "2-STAR", "3-STAR", "4-STAR", "5-STAR"]].sum(axis = 1)
WordSetSummary.style.set_precision(0)


# In[26]:


# BABY PACIFIER HELPFULNESS RATING
BPGroup = DataSet[DataSet["PRODUCT CATEGORY"] == "Baby"]
BPHelpfulVotes = BPGroup.groupby(by = "STAR RATING")["HELPFUL VOTES"].sum().to_frame("HELPFUL VOTES")
BPTotalVotes = BPGroup.groupby(by = "STAR RATING")["TOTAL VOTES"].sum().to_frame("TOTAL VOTES")
BPHelpfulVotes["TOTAL VOTES"] = BPTotalVotes["TOTAL VOTES"].tolist()
BPHV = np.array(BPHelpfulVotes["HELPFUL VOTES"].tolist())
BPTV = np.array(BPHelpfulVotes["TOTAL VOTES"].tolist())
Ratio = np.round(BPHV/BPTV*100, decimals = 2)
BPRatio = [f"{i}%" for i in Ratio]
BPHelpfulVotes["RATIO"] = BPRatio
BPHelpfulVotes.T


# In[27]:


# HAIR DRYER HELPFULNESS RATING
HDGroup = DataSet[DataSet["PRODUCT CATEGORY"] == "Beauty"]
HDHelpfulVotes = HDGroup.groupby(by = "STAR RATING")["HELPFUL VOTES"].sum().to_frame("HELPFUL VOTES")
HDTotalVotes = HDGroup.groupby(by = "STAR RATING")["TOTAL VOTES"].sum().to_frame("TOTAL VOTES")
HDHelpfulVotes["TOTAL VOTES"] = HDTotalVotes["TOTAL VOTES"].tolist()
HDHV = np.array(HDHelpfulVotes["HELPFUL VOTES"].tolist())
HDTV = np.array(HDHelpfulVotes["TOTAL VOTES"].tolist())
Ratio = np.round(HDHV/HDTV*100, decimals = 2)
HDRatio = [f"{i}%" for i in Ratio]
HDHelpfulVotes["RATIO"] = HDRatio
HDHelpfulVotes.T


# In[28]:


# MICROWAVE OVEN HELPFULNESS RATING
MOGroup = DataSet[DataSet["PRODUCT CATEGORY"] == "Major Appliances"]
MOHelpfulVotes = MOGroup.groupby(by = "STAR RATING")["HELPFUL VOTES"].sum().to_frame("HELPFUL VOTES")
MOTotalVotes = MOGroup.groupby(by = "STAR RATING")["TOTAL VOTES"].sum().to_frame("TOTAL VOTES")
MOHelpfulVotes["TOTAL VOTES"] = MOTotalVotes["TOTAL VOTES"].tolist()
MOHV = np.array(MOHelpfulVotes["HELPFUL VOTES"].tolist())
MOTV = np.array(MOHelpfulVotes["TOTAL VOTES"].tolist())
Ratio = np.round(MOHV/MOTV*100, decimals = 2)
MORatio = [f"{i}%" for i in Ratio]
MOHelpfulVotes["RATIO"] = MORatio
MOHelpfulVotes.T


# In[29]:


# BABY PACIFIER SENTIMENT INTENSITY ANALYZER
BabyPacifierSentiment = {}
for i, row in tqdm(BabyPacifier.iterrows(), total = len(BabyPacifier)):
    Reviews = row["REVIEW BODY"]
    ID = row["CUSTOMER ID"]
    BabyPacifierSentiment[ID] = SentimentIntensityAnalyzer().polarity_scores(Reviews)
BPSentiment = pd.DataFrame(BabyPacifierSentiment).T
BPSentiment = BPSentiment.reset_index().rename(columns = {"index": "CUSTOMER ID"})
BPSentiment = BPSentiment.merge(BabyPacifier, how = "left")
BPSentiment.index = BPSentiment.index + 1
BPSentiment[["neg", "neu", "pos", "STAR RATING"]]


# In[30]:


# HAIR DRYER SENTIMENT INTENSITY ANALYZER
HairDryerSentiment = {}
for i, row in tqdm(HairDryer.iterrows(), total = len(HairDryer)):
    HDReviews = row["REVIEW BODY"]
    HDID = row["CUSTOMER ID"]
    HairDryerSentiment[HDID] = SentimentIntensityAnalyzer().polarity_scores(HDReviews)
HDSentiment = pd.DataFrame(HairDryerSentiment).T
HDSentiment = HDSentiment.reset_index().rename(columns = {"index": "CUSTOMER ID"})
HDSentiment = HDSentiment.merge(HairDryer, how = "left")
HDSentiment.index = HDSentiment.index + 1
HDSentiment[["neg", "neu", "pos", "STAR RATING"]]


# In[31]:


# MICROWAVE OVEN SENTIMENT INTENSITY ANALYZER
MicrowaveOvenSentiment = {}
for i, row in tqdm(MicrowaveOven.iterrows(), total = len(MicrowaveOven)):
    MOReviews = row["REVIEW BODY"]
    MOID = row["CUSTOMER ID"]
    MicrowaveOvenSentiment[MOID] = SentimentIntensityAnalyzer().polarity_scores(MOReviews)
MOSentiment = pd.DataFrame(MicrowaveOvenSentiment).T
MOSentiment = MOSentiment.reset_index().rename(columns = {"index": "CUSTOMER ID"})
MOSentiment = MOSentiment.merge(MicrowaveOven, how = "left")
MOSentiment.index = MOSentiment.index + 1
MOSentiment[["neg", "neu", "pos", "STAR RATING"]]


# In[33]:


# SENTIMENT INTENSITY ANALYZER EXCEL SHEET
BPSentiment[["neg", "neu", "pos", "STAR RATING"]].to_excel("BPSentiment.xlsx")
HDSentiment[["neg", "neu", "pos", "STAR RATING"]].to_excel("HDSentiment.xlsx")
MOSentiment[["neg", "neu", "pos", "STAR RATING"]].to_excel("MOSentiment.xlsx")  


# In[105]:


# TIME-SERIES DATAFRAME
BPSeries = BabyPacifier["REVIEW DATE"].unique()
HDSeries = HairDryer["REVIEW DATE"].unique()
MOSeries = MicrowaveOven["REVIEW DATE"].unique()
BPSet = set(BPSeries.tolist())
HDSet = set(HDSeries.tolist())
MOSet = set(MOSeries.tolist())
SeriesIntersection = list(HDSet.intersection(BPSet, MOSet))
DateIntersections = pd.DataFrame(SeriesIntersection, columns = ["DATE INTERSECTIONS"])
DateIntersections.sort_values(by = "DATE INTERSECTIONS", ascending = False)
DateIntersections.to_excel("DateIntersections.xlsx")
TimeSeriesDataFrame = pd.read_csv("TimeSeriesAnalysis.csv")
TimeSeriesDataFrame.index = TimeSeriesDataFrame.index + 1
TimeSeriesDataFrame


# In[170]:


# COMBINED TIME-SERIES VISUALIZATION
ColorPalette = ["orange", "purple", "green"]
TimeSeriesDataFrame["REVIEW DATE"] = pd.to_datetime(TimeSeriesDataFrame["REVIEW DATE"])
TimeSeries = TimeSeriesDataFrame.plot("REVIEW DATE", figsize = (14.5, 4), marker = "o", color = ColorPalette, lw = 1, ms = 5)
plt.xlabel("\nREVIEW DATE")
plt.ylabel("PRODUCT PURCHASES")
plt.grid(alpha = 0.3)


# In[169]:


# INDIVIDUAL TIME-SERIES VISUALIZATION
DataFrame = TimeSeriesDataFrame.set_index("REVIEW DATE")
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize = (14.5, 11), sharex = True)
DataFrame.plot(subplots = True, ax = (ax1, ax2, ax3), marker = 'o', linewidth = 1, markersize = 5, color = ColorPalette)
ax1.grid(alpha = 0.3)
ax2.grid(alpha = 0.3)
ax3.grid(alpha = 0.3)
ax1.legend(loc = "upper left")
ax2.legend(loc = "upper left")
ax3.legend(loc = "upper left")
plt.xlabel("\nREVIEW DATE")
ax2.set_ylabel("PRODUCT PURCHASES\n")


# In[179]:


# TIME-SERIES DIFFERENCING
DataFrame["BABY PACIFIER DIFFERENCE"] = DataFrame["BABY PACIFIER"].diff()
DataFrame["HAIR DRYER DIFFERENCE"] = DataFrame["HAIR DRYER"].diff()
DataFrame["MICROWAVE OVEN DIFFERENCE"] = DataFrame["MICROWAVE OVEN"].diff()
Frame = ["BABY PACIFIER DIFFERENCE", "HAIR DRYER DIFFERENCE", "MICROWAVE OVEN DIFFERENCE"]
fig2, (ax11, ax22, ax33) = plt.subplots(3, 1, figsize = (14.5, 11), sharex = True)
DataFrame[Frame].plot(subplots = True, ax = (ax11, ax22, ax33), marker = 'o', linewidth = 1, markersize = 5,
                      color = ColorPalette)
ax11.grid(alpha = 0.3)
ax22.grid(alpha = 0.3)
ax33.grid(alpha = 0.3)
ax11.legend(loc = "upper left")
ax22.legend(loc = "upper left")
ax33.legend(loc = "upper left")
plt.xlabel("\nREVIEW DATE")


# In[206]:


# BABY PACIFIER CORRELOGRAM ANALYSIS
figBP, BPdimensions = plt.subplots(figsize = (10, 3), sharey = True)
pyplot.grid(True, alpha = 0.3)
plot_acf(DataFrame["BABY PACIFIER DIFFERENCE"].dropna(), color = "fuchsia", ax = BPdimensions,
         vlines_kwargs = {"colors": "fuchsia"})
plt.title("AUTOCORRELATION FUNCTION (BABY PACIFIER)\n", fontsize = 15)
for item in BPdimensions.collections:
    if type(item) == PolyCollection:
        item.set_facecolor("fuchsia")
figBP, BPdimensions = plt.subplots(figsize = (10, 3),  sharey = True)
pyplot.grid(True, alpha = 0.3)
plot_pacf(DataFrame["BABY PACIFIER DIFFERENCE"].dropna(), color = "fuchsia", ax = BPdimensions,
          vlines_kwargs = {"colors": "fuchsia"})
plt.title("PARTIAL AUTOCORRELATION FUNCTION (BABY PACIFIER)\n", fontsize = 15)
for item in BPdimensions.collections:
    if type(item) == PolyCollection:
        item.set_facecolor("fuchsia")


# In[207]:


# HAIR DRYER CORRELOGRAM ANALYSIS
figHD, HDdimensions = plt.subplots(figsize = (10, 3), sharey = True)
pyplot.grid(True, alpha = 0.3)
plot_acf(DataFrame["HAIR DRYER DIFFERENCE"].dropna(), color = "blue", ax = HDdimensions,
         vlines_kwargs = {"colors": "blue"})
plt.title("AUTOCORRELATION FUNCTION (HAIR DRYER)\n", fontsize = 15)
for item in HDdimensions.collections:
    if type(item) == PolyCollection:
        item.set_facecolor("blue")
figHD, HDdimensions = plt.subplots(figsize = (10, 3),  sharey = True)
pyplot.grid(True, alpha = 0.3)
plot_pacf(DataFrame["HAIR DRYER DIFFERENCE"].dropna(), color = "blue", ax = HDdimensions,
          vlines_kwargs = {"colors": "blue"})
plt.title("PARTIAL AUTOCORRELATION FUNCTION (HAIR DRYER)\n", fontsize = 15)
for item in HDdimensions.collections:
    if type(item) == PolyCollection:
        item.set_facecolor("blue")


# In[208]:


# MICROWAVE OVEN CORRELOGRAM ANALYSIS
figMO, MOdimensions = plt.subplots(figsize = (10, 3), sharey = True)
pyplot.grid(True, alpha = 0.3)
plot_acf(DataFrame["MICROWAVE OVEN DIFFERENCE"].dropna(), color = "red", ax = MOdimensions,
         vlines_kwargs = {"colors": "red"})
plt.title("AUTOCORRELATION FUNCTION (MICROWAVE OVEN)\n", fontsize = 15)
for item in MOdimensions.collections:
    if type(item) == PolyCollection:
        item.set_facecolor("red")
figMO, MOdimensions = plt.subplots(figsize = (10, 3),  sharey = True)
pyplot.grid(True, alpha = 0.3)
plot_pacf(DataFrame["MICROWAVE OVEN DIFFERENCE"].dropna(), color = "red", ax = MOdimensions,
          vlines_kwargs = {"colors": "red"})
plt.title("PARTIAL AUTOCORRELATION FUNCTION (MICROWAVE OVEN)\n", fontsize = 15)
for item in MOdimensions.collections:
    if type(item) == PolyCollection:
        item.set_facecolor("red")


# In[213]:


# BABY PACIFIER MODEL FITTING
StepwiseFit = auto_arima(TimeSeriesDataFrame["BABY PACIFIER"], trace = True, suppress_warnings = True)


# In[214]:


# HAIR DRYER MODEL FITTING
StepwiseFit = auto_arima(TimeSeriesDataFrame["HAIR DRYER"], trace = True, suppress_warnings = True)


# In[215]:


# MICROWAVE OVEN MODEL FITTING
StepwiseFit = auto_arima(TimeSeriesDataFrame["MICROWAVE OVEN"], trace = True, suppress_warnings = True)


# In[221]:


# TRAINING THE MODEL FIT
TrainData = TimeSeriesDataFrame.iloc[:-10]
TestData = TimeSeriesDataFrame.iloc[-10:]
print("TRAIN DATA:", TrainData.shape, "\n", "TEST DATA:", TestData.shape)


# In[224]:


# BABY PACIFIER ARIMA MODEL
BPARIMAmodel = ARIMA(TrainData["BABY PACIFIER"], order = (2,1,1))
BPARIMAmodel = BPARIMAmodel.fit()
BPARIMAmodel.summary()


# In[226]:


# HAIR DRYER ARIMA MODEL
HDARIMAmodel = ARIMA(TrainData["HAIR DRYER"], order = (0,1,1))
HDARIMAmodel = HDARIMAmodel.fit()
HDARIMAmodel.summary()


# In[227]:


# MICROWAVE OVEN ARIMA MODEL
MOARIMAmodel = ARIMA(TrainData["MICROWAVE OVEN"], order = (0,1,1))
MOARIMAmodel = MOARIMAmodel.fit()
MOARIMAmodel.summary()


# In[259]:


# BABY PACIFIER MODEL ACCURACY TEST
BPPredictionAccuracy = BPARIMAmodel.predict(start = len(TrainData) + 1, end = len(TrainData) + len(TestData))
BPMAPEvalue = mean_absolute_percentage_error(TestData["BABY PACIFIER"], BPPredictionAccuracy)
BPMAPE = "{:.2%}".format(BPMAPEvalue*0.1)
print("MEAN ABSOLUTE PERCENTAGE ERROR (BABY PACIFIER):", BPMAPE)


# In[260]:


# HAIR DRYER MODEL ACCURACY TEST
HDPredictionAccuracy = HDARIMAmodel.predict(start = len(TrainData) + 1, end = len(TrainData) + len(TestData))
HDMAPEvalue = mean_absolute_percentage_error(TestData["HAIR DRYER"], HDPredictionAccuracy)
HDMAPE = "{:.2%}".format(HDMAPEvalue*0.1)
print("MEAN ABSOLUTE PERCENTAGE ERROR (HAIR DRYER):", HDMAPE)


# In[258]:


# MICROWAVE OVEN MODEL ACCURACY TEST
MOPredictionAccuracy = MOARIMAmodel.predict(start = len(TrainData) + 1, end = len(TrainData) + len(TestData))
MOMAPEvalue = mean_absolute_percentage_error(TestData["HAIR DRYER"], MOPredictionAccuracy)
MOMAPE = "{:.2%}".format(MOMAPEvalue*0.1)
print("MEAN ABSOLUTE PERCENTAGE ERROR (MICROWAVE OVEN):", MOMAPE)


# In[293]:


# BABY PACIFIER ARIMA FORECAST VISUALIZATION
BPTotalForecast = BPARIMAmodel.predict(start = len(TimeSeriesDataFrame) - 765, end = len(TimeSeriesDataFrame) + 10)
plt.plot(TimeSeriesDataFrame["BABY PACIFIER"], color = "crimson", label = "ACTUAL BABY PACIFIER PURCHASES", alpha = 0.5)
BPFuturePrediction = BPTotalForecast.plot(figsize = (14.5, 2.5), color = "crimson", label = "FORECASTED BABY PACIFIER PURCHASES")
plt.legend(loc = "upper left")
BPFuturePrediction.set_xlabel("\nDATE INDEX")
BPFuturePrediction.set_ylabel("PRODUCT PURCHASES\n")
BPFuturePrediction.grid(True, alpha = 0.3)


# In[294]:


# HAIR DRYER ARIMA FORECAST VISUALIZATION
HDTotalForecast = HDARIMAmodel.predict(start = len(TimeSeriesDataFrame) - 765, end = len(TimeSeriesDataFrame) + 10)
plt.plot(TimeSeriesDataFrame["HAIR DRYER"], color = "teal", label = "ACTUAL HAIR DRYER PURCHASES", alpha = 0.5)
HDFuturePrediction = HDTotalForecast.plot(figsize = (14.5, 2.5), color = "teal", label = "FORECASTED HAIR DRYER PURCHASES")
plt.legend(loc = "upper left")
HDFuturePrediction.set_xlabel("\nDATE INDEX")
HDFuturePrediction.set_ylabel("PRODUCT PURCHASES\n")
HDFuturePrediction.grid(True, alpha = 0.3)


# In[295]:


# MICROWAVE OVEN ARIMA FORECAST VISUALIZATION
MOTotalForecast = MOARIMAmodel.predict(start = len(TimeSeriesDataFrame) - 765, end = len(TimeSeriesDataFrame) + 10)
plt.plot(TimeSeriesDataFrame["MICROWAVE OVEN"], color = "darkorchid", label = "ACTUAL MICROWAVE OVEN PURCHASES", alpha = 0.5)
MOFuturePrediction = MOTotalForecast.plot(figsize = (14.5, 2.5), color = "darkorchid", label = "FORECASTED MICROWAVE OVEN PURCHASES")
plt.legend(loc = "upper left")
MOFuturePrediction.set_xlabel("\nDATE INDEX")
MOFuturePrediction.set_ylabel("PRODUCT PURCHASES\n")
MOFuturePrediction.grid(True, alpha = 0.3)


# ![FOOTER-2.png](attachment:FOOTER-2.png)
