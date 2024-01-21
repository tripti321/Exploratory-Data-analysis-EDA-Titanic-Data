{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "374b4def",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "16de49bb",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "id": "da848252",
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "a90422ff",
   "metadata": {},
   "outputs": [],
   "source": [
    "import seaborn as sns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "id": "9abc4b03",
   "metadata": {},
   "outputs": [],
   "source": [
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "id": "548adfe7",
   "metadata": {},
   "outputs": [],
   "source": [
    "train = pd.read_csv('titanic_train.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "id": "e0f88749",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Braund, Mr. Owen Harris</td>\n",
       "      <td>male</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>A/5 21171</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>\n",
       "      <td>female</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>PC 17599</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C85</td>\n",
       "      <td>C</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>Heikkinen, Miss. Laina</td>\n",
       "      <td>female</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>STON/O2. 3101282</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>\n",
       "      <td>female</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>113803</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>C123</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Allen, Mr. William Henry</td>\n",
       "      <td>male</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>373450</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   PassengerId  Survived  Pclass  \\\n",
       "0            1         0       3   \n",
       "1            2         1       1   \n",
       "2            3         1       3   \n",
       "3            4         1       1   \n",
       "4            5         0       3   \n",
       "\n",
       "                                                Name     Sex   Age  SibSp  \\\n",
       "0                            Braund, Mr. Owen Harris    male  22.0      1   \n",
       "1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   \n",
       "2                             Heikkinen, Miss. Laina  female  26.0      0   \n",
       "3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   \n",
       "4                           Allen, Mr. William Henry    male  35.0      0   \n",
       "\n",
       "   Parch            Ticket     Fare Cabin Embarked  \n",
       "0      0         A/5 21171   7.2500   NaN        S  \n",
       "1      0          PC 17599  71.2833   C85        C  \n",
       "2      0  STON/O2. 3101282   7.9250   NaN        S  \n",
       "3      0            113803  53.1000  C123        S  \n",
       "4      0            373450   8.0500   NaN        S  "
      ]
     },
     "execution_count": 54,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "29713893",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Exploratory Data Analaysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "a588676a",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Missing data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "id": "f3a9ede2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>886</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>887</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>888</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>889</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>890</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>891 rows × 12 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     PassengerId  Survived  Pclass   Name    Sex    Age  SibSp  Parch  Ticket  \\\n",
       "0          False     False   False  False  False  False  False  False   False   \n",
       "1          False     False   False  False  False  False  False  False   False   \n",
       "2          False     False   False  False  False  False  False  False   False   \n",
       "3          False     False   False  False  False  False  False  False   False   \n",
       "4          False     False   False  False  False  False  False  False   False   \n",
       "..           ...       ...     ...    ...    ...    ...    ...    ...     ...   \n",
       "886        False     False   False  False  False  False  False  False   False   \n",
       "887        False     False   False  False  False  False  False  False   False   \n",
       "888        False     False   False  False  False   True  False  False   False   \n",
       "889        False     False   False  False  False  False  False  False   False   \n",
       "890        False     False   False  False  False  False  False  False   False   \n",
       "\n",
       "      Fare  Cabin  Embarked  \n",
       "0    False   True     False  \n",
       "1    False  False     False  \n",
       "2    False   True     False  \n",
       "3    False  False     False  \n",
       "4    False   True     False  \n",
       "..     ...    ...       ...  \n",
       "886  False   True     False  \n",
       "887  False  False     False  \n",
       "888  False   True     False  \n",
       "889  False  False     False  \n",
       "890  False   True     False  \n",
       "\n",
       "[891 rows x 12 columns]"
      ]
     },
     "execution_count": 55,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train.isnull()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "id": "64663e4c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: >"
      ]
     },
     "execution_count": 56,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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zZvnXFEZGDt2rGbMmKG8efMqJCTE83lQUJCjYeChhx7Sp59+qqSkJHXs2FEzZ870+n7WrFn12GOPOVRdxqKiohQZGakXXnjBc2Nbt26dgoKC9N577zlc3XXr16/Xxx9/rBIlSjhdyk1eeeUVSQY9gFo2evrpp61Tp055fXb69GkrPDzczjJuq3LlytalS5e8Prt8+bJVsWJFhyq6WXot0dHRVunSpa2UlBTLsiyrdOnSTpZ1k/DwcCsxMdGyLMsqX768ZVmWdfbsWaP+fx8+fNh67rnnrHr16lkzZ860ypYtaw0cONC6ePGi06VZlmVZnTp1ssaMGWNZlmVVqFDBsizLmjVrltWhQwcny7pJ06ZNrR07dnh9Fh0dbTVv3tyhijL25JNPWtu2bXO6jNuKiYlxuoS/pGHDhtbevXsty/r9+v7555+tatWqOViVtyeffNJKTU11uozbmjVrlvXee+9ZSUlJlmVZVnJysjVgwADrww8/tLUOWx8vEhMTde+993p9ljVrVp0/f97OMm4rLS3tpkVux48f98zPmyB79uxKSEjQunXrVK5cOQUHB+vw4cPKnTu306V5qVWrlnr27KlffvlFQUFBSkhI0ODBgxUeHu50aR7NmjVTWFiYPvvsM7Vv316ff/65oqOj1bhxY6dLkyT169dPUVFRql69upKTk9WgQQN9/PHH6t27t9OlealUqZI6d+6s4cOHa+HChRo7dqzat2+vggULatKkSZ4/TsucObMqVarkdBm3VahQIS1ZskSNGjVSpUqVFBsbq65duyo5Odnp0rycOnVKpUqVkvT70+1DDz2kixcvOlmWl4YNG2r27NlOl3Fbs2fP1uDBg3XPPfdIku666y7169dPS5YssbUOWyd2KlSooF69eqlnz54qUKCAjh07phEjRhh1c3juuefUrl07z0B27NgxzZo1Sy+99JLTpXk0a9ZMTZo0UVJSkiZMmKCDBw+qffv2atu2rdOleenRo4f69OmjevXqSZKqVq2q8PBwo9rwI0eOVIMGDTxfP/jgg1q8eLEmTJjgYFW/K1SokL744gutX79esbGxKlCggGrUqKG7777b6dK8HDx4UE888YQOHTrkWf9TuHBhJSQkeML1n60bskPNmjW1atUqNWzY0OlSbmnevHlavHix2rVrp5EjRypHjhyKi4tTZGSkhg4d6nR5Hg8//LC++eYbPf30057Pvv32Wz300EMOVuUtOjpae/bs0dSpU29aA/TNN984VJW39AfQBx54wPOZEw+gtu4miI+PV7du3bR7927PwFClShWNGTPmpo6BU1JTUzV58mStXLlScXFxKliwoF544QV16NDBiMEs3fbt25U1a1aVLl1aJ0+e1IEDB1S3bl2ny8pQYmKijh8/rgIFCui+++5zupwMff/99zp+/Lhq1Kih8+fPKzQ01OmSJEmxsbEZfh4SEqKcOXMqS5YsNlfkn1555RUFBQUpOTlZhw4d0j//+U/lypXL6+84vaAs3TPPPKMpU6aocOHCnkWjp0+fVtOmTY3Y4ZLu22+/1euvv67atWtr7dq1atq0qVatWqXRo0cb84D3+eef3/J7TZs2tbGSW4uMjNTGjRtvegBt3LixunbtalsdjmwtPHHihE6fPq0CBQqoYMGCdv/zAeHMmTPKmzevrl69qqVLlyp37tyqX7++02XdZNeuXTpx4oT++Gvm5GKtG5m+/TEsLOymLVHpMmXKpKeeekoffPCBozsfLly4oPj4eD3yyCOSpGXLlunQoUOqU6eOMS35vzJFYcpWs4oVK2rbtm3KlCmTKlSooJ07d+ratWt66qmnjNpBIl1feP3JJ5/oxIkTKlCggCIiIozb4mw6Ux5AOWcgA1u2bNGCBQsUFxdn5HaZTz/9VMOGDdO+ffs0bNgwrV69WkFBQWrZsqVef/11p8vzGDRokJYuXar77rvP65c6KCjImBad6dsfFyxYoPXr16tv374qVKiQjh8/rpEjR6p48eKqW7eupk6dquDgYI0aNcqR+o4ePapXXnlFNWvW1LBhwzRv3jyNHj1aNWvW1Pbt2zV69GhVrVrVkdpu5ejRo8qfP7/uvvtu7d27V/fee68KFy7sdFkerVu3Vv369dWiRQtPZyAqKkqffPKJFixY4HR5Hl26dNGoUaOMm7KSpI4dO2rGjBmejlBGTOkEGcOOVYqPP/64VbRo0dv+McXKlSutypUrW2PGjLHKli1rnT592qpbt671wQcfOF2aR+PGja3NmzdbqampVtmyZa3du3dbMTExRq3St6zrK4wPHDjgdBm39dRTT3l2DqSv1r969apndbTTnn76aevs2bNen/32229W7dq1LcuyrPPnzzu60+Wtt96yhg0b5lmxXa1aNWv27NmWZVnWhg0b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pqaKjY3VO++8owEDBigsLEwffPCB9u3bp4kTJypv3rxOl+clLi5OefLkUUhIiFavXq0LFy6oSZMmfrF3Hn+uYcOGWrVqlWrVqpXh94OCgozaP9+tWzdly5ZNAwYMUNu2bdWkSRNly5ZNkyZNcrzOU6dOqUCBAjedRHgjE04gLFmypNeLvAYPHnzTS9JMCICWZenYsWPKnz+/QkJClClTJk2aNEllypQxalpo0KBBOnDggJ555hndd999Xl0LO3+OtoaB8ePHKzY2VgMHDlSOHDl08eJFRUZG6v7771fr1q01fvx4xcTEONoOK1WqlB5++GG98MILaty48U2v3jVBp06dFBoaqr59++ruu+9WYmKixo4dq3Pnzv3pVALw35aWlqbffvvNc4jT1q1bdfjwYYWHhxt14JB0/Smsf//+Gjp0qGJiYry2w6W/CdQpYWFhio6OvmlhmWTW2z5vFfzSmRAAL168qLZt2ypv3ryeRa4JCQmqWbOmihcvrlmzZhlziJwpQdrWMFCzZk2tXr3aa9X7pUuXVL9+fW3YsEFXrlxRtWrVHF2Zev78eUVFRWn58uX63//9Xz399NOKiIgwar90xYoVtWXLFoWEhHg+u3LliqpXr27Ei4Fq1ap12zm5oKAgrV271saK4Cv+9MKnSZMmKTo6WlWrVvXs4zZpJ06xYsV06NCh2y5uSz+YCLc3evRo7du3T+PGjVNoaKjn84SEBHXp0kWVK1c26t0EJrB1EurixYtKSkryuvDOnz+vCxcueL52emHHPffco5YtW6ply5Y6evSoVq5cqT59+igkJERff/21o7WlCw4OVmJiotfWx3PnzhlzDvxbb72V4ef79u3TJ598oieeeMLmiuArY8eO1eOPP653331XkjRx4kR16NDB88KniRMnOv7CJ+n66afLly9X+fLlNWHCBM/77IODg42Zi08fF7nh//9bs2aNZs6c6RUEpOtrRgYPHqxu3boZFQaOHTumuLg4z86M9LfQvvbaa7bVYOtVUK9ePb3xxht65513dP/99ys2NlYTJkxQ3bp1deHCBQ0dOlTly5e3s6Rbunjxor777jsdOHBA586d+9PWmJ3q1aunrl27qlu3bipYsKBOnjypCRMm6JlnnnG6NEkZH0M8Z84cLVu2TC1atFCfPn0cqAq+sGXLFq8XPsXExKhx48aSrq8RGjp0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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "06623dae",
   "metadata": {},
   "outputs": [],
   "source": [
    "# the propation of age missing is likely small enough for reasonable replacement with some impulation. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "id": "87fd1206",
   "metadata": {},
   "outputs": [],
   "source": [
    "sns.set_style('whitegrid')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "id": "d8ff6dd5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='Survived', ylabel='count'>"
      ]
     },
     "execution_count": 58,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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IGwAAIAxCBsAAGAMwgYAABiDsAEAAMbwOmyys7PPuHzp0qUXPQwAAMDFCBrKRp9++qlefvllSdKePXv0zDPPeKzv7OxUc3PzsA8HAADgjSGFzfe+9z21tLTo2LFjcrvdqqur81h/6aWXKi8vzycDAgAADNWQwiYwMFC///3vJUm5ublat27dRb3o22+/rY0bN+rgwYOyWq26/fbblZOTo5CQEDU0NGjdunVqbW1VWFiYsrOzlZGRMbhvVVWVNm/erC+//FLXXHONnnjiCcXGxl7UPAAAwAxef8Zm3bp16u3t1RdffKHDhw97/BmKY8eO6Ze//KV++tOfat++faqqqtI777yj5557Th0dHVqxYoVSU1NVX1+vgoICbdiwQQcOHJAk1dXVae3atSosLFR9fb2Sk5OVnZ2tnp4eb08DAAAYaEh3bL7t1Vdf1RNPPKHOzs7BZQMDAwoICNC///3v8+4fHh6ut956S5dddpkGBgZ04sQJff311woPD1dNTY1sNpsyMzMlSfHx8UpKSlJ5ebliYmJUUVGhxMRExcXFSZKysrK0bds27dq1S4sWLfLqPNxut1fbe8Nisfjs2MBY58trbyRwfQNn58vre6jH9jpsioqKlJmZqbvuuktBQV7vLkm67LLLJEk333yzjhw5ojlz5igtLU2bNm1SZGSkx7Z2u12VlZWSpNbW1tMCxm63q6mpyesZGhsbL2j287FarZoxY4ZPjg2YoLm5eczeZeX6Bs5tNFzfXpfJ559/rvvvv/+Co+bbampq1NHRoVWrVumBBx7QVVddJavV6rFNSEiIuru7JUldXV3nXO+N6OhofvMC/CAqKsrfIwDwEV9e3263e0g3Jbyuk5kzZ6q1tVXTp0+/oMG+LSQkRCEhIcrJyVFGRoaWLVumkydPemzjdDoVGhoq6ZvflpxO52nrw8LCvH5ti8VC2AB+wHUHmGs0XN9eh83s2bOVlZWl22+/XRMnTvRYd//99593//fee09r1qzRjh07FBwcLEnq7e3VJZdcIrvdrr1793ps39raqoiICElSRESEWlpaTlt/0003eXsaAADAQF5/K2r//v2KiIjQwYMHVVdXN/jnnXfeGdL+UVFRcjqd+t3vfqfe3l599tln+vWvf6309HTddtttamtrU1lZmVwul2pra1VdXT34uZr09HRVV1ertrZWLpdLZWVl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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.countplot(x='Survived',data =train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "id": "39030a20",
   "metadata": {},
   "outputs": [],
   "source": [
    "sns.set_style('whitegrid')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "id": "9684e011",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='Survived', ylabel='count'>"
      ]
     },
     "execution_count": 60,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAjYAAAGsCAYAAADOo+2NAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8pXeV/AAAACXBIWXMAAA9hAAAPYQGoP6dpAAAoUUlEQVR4nO3de3xU9Z3/8ffMhJDhmkTDxRpv5GKRQICgUEAhNO7aLCFCgrsLrPFSu4BiUUClVHgAgVR7QXSxCoasW3xgSQ0QGzWtRSsoaWC5VSRNWOSmQklCyG2cyWR+f7jkZ5ZbJmQyk29ez8eDx8PMOXPmc5JHJi/PnJlj8Xg8HgEAABjA6u8BAAAA2gphAwAAjEHYAAAAYxA2AADAGIQNAAAwBmEDAACMQdgAAABjBPl7gPbW2NiohoYGWa1WWSwWf48DAABawOPxqLGxUUFBQbJaL31cptOFTUNDgw4cOODvMQAAQCvExcUpODj4kss7Xdicr7y4uDjZbDY/TwMAAFrC7XbrwIEDlz1aI3XCsDn/8pPNZiNsAADoYK50GgknDwMAAGMQNgAAwBiEDQAAMEanO8cGAABvNTY2yul0+nsMo3Xp0qVNzn0lbAAAuAyn06kjR46osbHR36MYLzQ0VP369buqz5kjbAAAuASPx6Mvv/xSNptNkZGRV3yrMVrH4/Gorq5Op0+fliT179+/1dsibAAAuISGhgbV1dXpuuuuU7du3fw9jtHsdrsk6fTp0+rTp0+rX5YiPQEAuAS32y1Jl/2kW7Sd8/HocrlavQ3CBgCAK+Dagu2jLb7PhA0AADAGYQMAAIzBycMAAHQQVVVV+tWvfqVt27apqqpKPXr00OjRozV37lz169fP3+MFBI7YAADQQcydO1eVlZXKzc3V3r17tXnzZjmdTj3wwANqaGjw93gBgbABAKCD2L17t5KSkhQRESFJuvbaa7Vw4UINGTJE586dU01NjZYuXaq77rpLo0aN0ty5c3XmzBlJ0u9//3sNGjRIhw4dkiQdPHhQgwcP1p///Ge/7Y8vEDYAAHQQycnJWrx4sZYsWaKCggKdPHlSERERysrKUnh4uBYuXKijR4/qrbfe0h//+Ef16NFDjz76qDwej5KTkzVx4kQtWLBAVVVVmjt3rjIyMnTnnXf6e7falMXj8Xj8PUR7crvd2rt3r+Lj49vkmhQX4/F4eGtgAOHnAaC1HA6Hjhw5optvvlkhISH+HkeNjY3Kz89XQUGBdu/ererqat1www167LHHNHr0aH3ve9/TO++8o1tuuUWSVF9fr4SEBL355psaNGiQ6urqNHnyZDmdTl133XX6z//8T5/9LWyNy32/W/r3m5OHfcBisWjXkVOqdnDBNH/rGRKshJv7+nsMAGgTVqtVkyZN0qRJk+TxeHT48GFt2bJFCxYs0BNPPCFJmjp1arP72Gw2nThxQoMGDVK3bt00ZcoU/fznP9fs2bMDKmraCmHjI9UOp6rqCRsAQNv46KOPNGfOHG3btk2hoaGyWCyKiorSk08+qR07djRdffydd95pOgdHksrKyhQZGSlJOnbsmF5++WWlp6frueee0+jRo417NxXn2AAA0AGMGDFC11xzjZ555hmVlJTI5XKppqZGW7du1eeff6577rlH48aNU2ZmpiorK+VyufTyyy8rLS1N586dk8vl0hNPPKHk5GQtX75cI0aM0Pz58427ajlhAwBABxASEqI33nhDERERmjlzphISEjRu3Dht3bpV69ev14ABA/Tcc8+pV69eSk1N1ciRI/Xhhx9q3bp1ioiI0AsvvKDKyko9/fTTkqSlS5eqrKxMr7zyip/3rG1x8rCPbPvsOC9FBYDe9mCN/26kv8cA0EEF2snDpmuLk4c5YgMAAIxB2AAAAGMQNgAAwBiEDQAAMAZhAwAAjEHYAAAAYxA2AADAGIQNAAAwBmEDAICX2vuzbTvZZ+leFS6CCQCAlywWi3YdOaVqh+8/Yb5nSLASbu7r88e5lMTERD366KOaPHmy32bwBmEDAEArVDucXDonAPFSFAAAhjlx4oRiY2O1efNmjR8/XvHx8XrmmWe0a9cupaSkaOjQobr//vtVUVGhmpoaLVq0SHfffbfi4+M1duxY/frXv77odp1Op1544QVNmDBBt99+u374wx/q6NGj7bx3l8cRGwAADPXhhx+qoKBAx48fV2pqqg4ePKi1a9eqS5cu+ud//me98cYbOnPmjE6cOKHc3Fz17NlThYWFmjNnju655x7deOONzbb3q1/9Sjt37lROTo769OmjtWvX6sEHH1RBQYG6du3qp71sjiM2AAAY6sEHH5TdbldMTIwiIiJ07733qm/fvgoPD1d8fLxOnjypxx57TKtWrVKPHj301VdfNQXK6dOnm23L4/Fo48aNeuKJJxQZGamuXbtq9uzZcrlc+uCDD/ywdxfHERsAAAwVGhra9N82m029evVq+tpqtcrj8ai8vFyZmZk6ePCgrr/+eg0aNEiS1NjY2GxbFRUVqqur0+OPPy6r9f8fF3G5XDp58qRvd8QLhA0AAIayWCxXXOfxxx9XYmKiXnvtNQUFBamyslK//e1vL1gvLCxMXbt2VXZ2tuLj45tu/5//+R/17eu/d239X7wUBQBAK/QMCVZvu+//9QwJ9ul+VFdXKyQkRDabTRUVFVq+fLmkb47EfJvValVaWpp+8Ytf6KuvvlJjY6Py8vL0T//0TwF1AjFHbAAA8JLH42nXz5bxeDwtOvrSGitXrtSKFSuUnZ2t3r176wc/+IEGDhyov/3tbxozZkyzdZ966im9+OKL+td//VedPXtWkZGRWr16tQYOHOiT2VrD4ulkH2fodru1d+9excfHy2az+exxtn12nM83CAC97cEa/91If48BoINyOBw6cuSIbr75ZoWEhPh7HONd7vvd0r/fvBQFAACMQdgAAABjEDYAAMAYhA0AADAGYQMAAIxB2AAAAGMQNgAAwBiEDQAAMAZhAwCAl9r7s2072WfpXhUuqQAAgJcsFouq//qxGmqrfP5YQd17q+eg7/n8cUxB2AAA0AoNtVVyV1f6e4yLKigo0LJly+R0OvXzn/9c48eP9/ljnjhxQhMmTND777+v66+/3uePdym8FAUAgGE2bdqk5ORk7d69u12iJpBwxAYAAIOkpaXp008/VXFxsT744ANlZ2drxYoV2rNnj7p166aUlBTNnj1bwcHBeuutt5Sbm6shQ4bod7/7naxWq2bPnq2uXbvq5Zdf1rlz55ScnKylS5dKkg4fPqznnntOJSUlqqio0PXXX6/58+dfNJ7OnDmjrKwsffLJJ7JYLEpMTNSCBQvUo0cPn+4/R2wAADBIbm6uEhIS9KMf/Uhbt25VRkaGoqOj9ec//1lvvPGGPv74Y7344otN6+/evVt9+/bVzp07NWfOHK1cuVJFRUUqKChQTk6OcnNzVVxcLEl67LHHFBMToz/84Q/atWuXxowZoyVLllwwQ2Njo2bNmiWr1ar33ntP+fn5On36tJ599lmf7z9hAwCAoT744AM5nU498cQT6tq1q/r376/HH39cGzZsaFqnW7duuv/++2W1WjVmzBi53W499NBDstvtiouLU58+fXTy5ElJ0iuvvKLHHntMHo9HJ0+eVK9evXTq1KkLHvevf/2rPv30Uy1evFg9evRQWFiYnnrqKf3+979XZaVvz0vipSgAAAx18uRJVVRUaMSIEU23eTweuVwulZeXS5JCQ0NlsVgkSVbrN8c7evXq1bS+1WpVY2OjJOnQoUOaNWuW/v73v2vAgAEKDw+/6FvRT5w4IbfbrbvuuqvZ7cHBwTp+/LjCwsLadke/hbABAMBQ/fr10w033KB333236baamhqVl5crPDxckpqi5kpOnTqlxx9/XC+99JISExMlSe+9954KCwsv+rghISEqKiqSzWaTJDmdTh0/flw33njj1e7WZfFSFAAArRDUvbdsPcN8/i+oe+9Wzzh+/HjV1tZq3bp1cjqdOnfunJ566inNnTu3xUFzXm1trdxut+x2uySprKxM//Ef/yHpm2j5tsGDB+vGG29UVlaWamtr5XA4tGLFCmVkZMjtdrd6f1qCIzYAAHjJ4/G064fmeTwer0NEknr06KGcnBxlZWVp3bp1amxs1B133KGXX37Z623dcsstWrBggebPn6/6+nr169dPU6dO1fPPP6+//e1vCg0NbVo3KChIr7zyin72s5/p7rvv1tdff63Bgwdr/fr16tq1q9eP7Q2Lp5N9TrPb7dbevXsVHx/fdHjMF7Z9dlxV9c4rrwif6m0P1vjvRvp7DAAdlMPh0JEjR3TzzTcrJCTE3+MY73Lf75b+/ealKAAAYAzCBgAAGMOvYeN2uzVjxgw9/fTTTbft27dP6enpGjp0qBITE7Vp06Zm98nLy1NSUpLi4+M1efJk7dmzp73HBgAAAcqvYfPSSy9p165dTV9XVVXpkUceUWpqqoqLi5WZmamVK1dq//79kqSioiItW7ZMWVlZKi4uVkpKimbOnKn6+np/7QIAAAggfgubTz75RIWFhbr77rubbissLFRoaKimTZumoKAgjRo1ShMnTmz6hMTzF/UaPny4unTpooyMDIWFhamgoMBfuwEA6AQ62fts/KYtvs9+ebt3eXm5fvKTn2jNmjXKyclpur20tFQxMTHN1o2KilJubq6kb94zP2XKlAuWHzp0yOsZfPk+el++2wqt4+vPTQBgLo/HI6fTybui2kFtba08Ho+sVusFz9stfR5v97BpbGzU/Pnz9cADD+jWW29ttqy2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AG95HTbvvvuufvrTn6qmpqbpNo/HI4vFos8++6xNhwMAAPCG12GzevVqTZs2Tffee6+Cgry+OwAAgM94XSZffvmlHn30UaIGAAAEHK8/x+a2225TWVmZL2YBAAC4Kl4fdhk2bJgyMjL0j//4j7r22mubLXv00UfbbDAAAABveR02e/bsUXR0tA4fPqzDhw833W6xWNp0MAAAAG95HTb/9V//5Ys5AAAArprXYXP+0goXk5qaehWjAAAAXJ1Wvd3726qqqlRfX6/hw4cTNgAAwK+8Dps//elPzb72eDxau3atzp4921YzAQAAtIrXb/f+vywWix566CFt2bKlLeYBAABotasOG0k6cuQI74oCAAB+5/VLUTNmzGgWMS6XSyUlJUpJSWnTwQAAALzlddjccccdzb62Wq3KyMjQ97///TYbCgAAoDW8Dptvf7pweXm5evfuzXWjAABAQPD6HBuXy6UVK1Zo6NChGjNmjIYPH66f/vSncjqdvpgPAACgxbwOmzVr1qioqEirVq3S22+/rVWrVmnfvn1atWqVD8YDAABoOa9fQ8rPz9f69esVGRkpSRowYIAGDBigadOmacGCBW0+IAAAQEt5fcSmqqpK/fv3b3Zb//795XA42mwoAABawuPx+HsE/K9A+Vl4fcQmNjZWGzdu1PTp05tu27hxo2JiYtp0MAAArsRisaj6rx+robbK36N0akHde6vnoO/5ewxJrQibH//4x3rwwQe1detWRUZG6tixYyorK9Nrr73mi/kAALishtoquasr/T0GAoTXYZOQkKCf/OQn2rdvn4KCgjR+/HhNnTpVw4YN88V8AAAALdaqq3vn5eVp/fr1uummm/T+++9rxYoVqqqq0sMPP+yLGQEAAFrE65OHc3Nz9frrr+umm26SJE2YMEHr16/Xhg0b2no2AAAAr3gdNjU1NRd9V1RdXV2bDQUAANAaXofNbbfdpldffbXZbdnZ2br11lvbbCgAAIDW8Pocm6effloPPvigfvvb36pfv3766quv1NDQoHXr1vliPgAAgBbzOmxuu+02FRYWatu2bTp9+rT69++vcePGqWfPnr6YDwAAoMVadVnu3r17KzU1tY1HAQAAuDpen2MDAAAQqAgbAABgDMIGAAAYg7ABAADGIGwAAIAxCBsAAGAMwgYAABiDsAEAAMYgbAAAgDEIGwAAYAzC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Rob+9Kc/6Xe/+50k6eTJk9q5c6cSEhKa1ne73brhhhskSadPn1b//v2blvXq1Uu9e/du1f4DaB+EDQCf++ijjzRnzhxt27ZNoaGhslgsioqK0pNPPqkdO3bo4MGDslqtcrlcTfeprKy8YDv/92hJWlqa3n77bR09elTf//73FRoaetHHT09P13PPPac+ffro1ltvVXR0tCSpX79+uvfee7V06dKmdU+fPi2Px9O0/Pjx403L6urqVF1d3ervAwDf46UoAD43YsQIXXPNNXrmmWdUUlIil8ulmpoabd26VZ9//rnGjRunAQMG6P3335fD4dDf//53vf7661fcbmpqatM7rNLT0y+53rhx41RXV6dXX3212Xrnw2j79u1qbGzU559/runTpys7O1vSN0G0bt06HT58WF9//bWysrLkdruv/hsCwGcIGwA+FxISojfeeEMRERGaOXOmEhISNG7cOG3dulXr16/XgAEDNG/ePNXW1mr06NH6t3/7N6WkpFxxu6GhoUpMTFRQUJBGjRp1yfWCgoI0efJkVVZW6p577mm6fciQIfrlL3+pX/7ylxoxYoSmT5+uxMREPfnkk5KkH/7wh0pJSdH06dM1ZswY9ezZ85JHhQAEBovn/DFXAACADo4jNgAAwBiEDQAAMAZhAwAAjEHYAAAAYxA2AADAGIQNAAAwBmEDAACMQdgAAABjEDYAAMAYhA0AADAGYQMAAIzx/wAvjLkBCyu7TgAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.countplot(x='Survived',hue='Sex' ,data =train ,palette ='RdBu_r')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "id": "0e2eb6f4",
   "metadata": {},
   "outputs": [],
   "source": [
    "sns.set_style('whitegrid')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "id": "b5561ab3",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='Survived', ylabel='count'>"
      ]
     },
     "execution_count": 62,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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AAAAjNWuj3v/tZqaGjU0NCgpKalNwaampkZTpkzRiBEjtHjxYtnt//fBrNTUVL300ktau3atJk2apP3792vLli1atWqVJCkjI0PTp0/X/fffr6SkJBUUFOj06dNKTU319TAAAICBfA42v//971t97fF4tHr1ap09e7ZN27/11luqqqrSO++8o+3bt7dqKykp0Zo1a7R48WKtXLlS0dHRWrBggUaMGCHpL5+Sys3N1cKFC3X8+HHFxMRo9erVioyM9PUwAACAgWwej8dzpTtxu90aNWrURdNIwcjtduvgwYNKSEhgrh8BdeYPK+SuOWZ1GZZxdLtRUaNmWl0GgE6qre/fPt+g71L+9Kc/tetTUQAAAB3J56moyZMntwoxTU1NKi8v1/jx4zu0MAAAAF/5HGxuv/32Vl/b7XZNnTpV99xzT4cVBQAA0B4+B5u/vrvw6dOn1a1bt3Y/DBMAAKAj+XyNTVNTk5YsWaLExESNHDlSSUlJeu6559TY2OiP+gAAANrM52CzatUqFRcXa8WKFdq6datWrFihQ4cOacWKFX4oDwAAoO18nkPasmWL3njjDd10002SpFtvvVW33nqrJk2apDlz5nR4gQAAAG3l84hNTU2N+vTp02pZnz59LnqGEwAAQKD5HGzi4uK0YcOGVss2bNigfv36dVhRAAAA7eHzVNTMmTP16KOPavPmzbrpppt09OhRVVZW6vXXX/dHfQAAAG3mc7AZNmyYfv7zn+vQoUMKCQnR3XffrQcffFBDhw71R30AAABt1q6nexcVFemNN97QLbfcovfee09LlixRTU2NsrOz/VEjAABAm/h8jU1hYaHWrVunW265RZI0ZswYvfHGGyooKOjo2gAAAHzic7Cpra295Kei6uvrO6woAACA9vA52AwcOFCvvfZaq2Vr1qxR//79O6woAACA9vD5Gpu5c+fq0Ucf1a9//Wv17t1bX3/9tZqbm5Wfn++P+gAAANrM52AzcOBA7dixQzt37tSJEyfUp08f3XXXXeratas/6gMAAGizdj2Wu1u3bpo4cWIHlwIAAHBlfL7GBgAAIFgRbAAAgDEINgAAwBgEGwAAYAyCDQAAMAbBBgAAGINgAwAAjEGwAQAAxiDYAAAAYxBsAACAMQg2AADAGAQbAABgDIINAAAwBsEGAAAYg2ADAACMQbABAADGINgAAABjEGwAAIAxCDYAAMAYlgab6upqpaamqri42LssNzdXgwYNUmJiove1ceNGb3tRUZFSU1OVkJCgtLQ0lZSUWFE6AAAIQiFWdbx//37NnTtXR48ebbW8tLRUeXl5euCBBy7apri4WHl5eVq9erUGDx6sgoIC5eTkaOfOnXI6nYEqHQAABClLRmyKioo0a9YsPfPMM62WNzY26rPPPtOgQYMuud2mTZs0duxYJSUlKTQ0VFOnTlVUVJS2bdsWiLIBAECQs2TEZuTIkRo3bpxCQkJahZuysjI1Nzdr5cqV2r9/v7p27ar09HRlZ2fLbrersrJS6enprfYVExOjsrIyn2twu91XfBxAWzkcDqtLCBqcewDao62/OywJNj179rzk8nPnzmn48OGaPHmyli9frk8//VTTp0+X3W5Xdna26urqLppyCg8PV319vc81lJaWtqt2wFdOp1MDBgywuoygUV5eroaGBqvLAGAoy66xuZSUlBSlpKR4vx48eLCmTJmibdu2KTs7W06nUy6Xq9U2LpdLUVFRPvcVHx/PX9GABeLi4qwuAUAn5Ha72zQoEVTB5ne/+51OnTqlhx56yLussbFR4eHhkqTY2FhVVFS02qayslKjRo3yuS+Hw0GwASzAeQfAn4LqPjYej0dLly7Vnj175PF4VFJSonXr1ikrK0uSlJGRoS1btmjv3r1qamrS2rVrdfr0aaWmplpcOQAACAZBNWKTmpqqefPmaeHChTp+/Lh69Oihp59+WhMmTJAkJScnKzc319seExOj1atXKzIy0trCAQBAULB5PB6P1UUEktvt1sGDB5WQkMCQOALqzB9WyF1zzOoyLOPodqOiRs20ugwAnVRb37+DaioKAADgShBsAACAMQg2AADAGAQbAABgDIINAAAwBsEGAAAYg2ADAACMQbABAB+4r65bf10W3wcEq6C68zAABDuHzaZ//OBzVda4vn9lQ8V0C9crf3uL1WUAl0SwAQAfVda49MczDVaXAeASmIoCAADGINgAAABjEGwAAIAxCDYAAMAYBBsAAGAMgg0AADAGwQYAABiDYAMAAIxBsAEAAMYg2AAAAGMQbAAAgDEINgAAwBgEGwAAYAyCDQAAV8DjabG6hKAQLN+HEKsLAACgM7PZ7Dp34E01156wuhTLhFzbS12HPmx1GZIINgAAXLHm2hNy1xyzugyIqSgAAGAQ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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.countplot(x='Survived' ,hue='Pclass',data =train,palette ='rainbow')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "id": "82b06712",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\tript\\anaconda3\\Lib\\site-packages\\seaborn\\axisgrid.py:118: UserWarning: The figure layout has changed to tight\n",
      "  self._figure.tight_layout(*args, **kwargs)\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<seaborn.axisgrid.FacetGrid at 0x1d07503af50>"
      ]
     },
     "execution_count": 63,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 500x500 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.displot(train['Age'].dropna(),kde=False,color='darkred',bins=40)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "id": "404562aa",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: >"
      ]
     },
     "execution_count": 64,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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e/dubd682c9NAABQMIYuwujln9ew4Nd2c7mLr998Cx3JZFJvvfWWGhoaFI/Hdfnll+vOO+/Uli1b/NoEAAAoYr4d92htbVVZWZkuvvji4WW/93u/p6NHj6q7u1ul4/xKx3VdSVIqlVI4HM67Htu2M+ZLp9NyLEuOx+NZoVBItm0X5FhXUiQelxsOZ6xTyDV7HXuuHgu5Zi9j3XB43P4KrWavY11J4WhUaccpmpq9jh35Oi2Gmr2MHe99OFk1S5IsS+l0WqlUyvtYafhng9f3Yl51+1RzLtt15d/P2SFDP2+Hfo6PJeRms1YWXnzxRT3xxBN64403hpcdPnxYtbW1+tnPfqZLLrlkzPGpVEotLS1+lAIAAAxLJBKKRqNjruPbbzqmTp2qvr6+jGVD30+bNm3c8ZFIRIlEQtan/0sAAACFz3VdOY6T1R+N+BY6Zs6cqZMnT+rEiRO66KKLJEmHDh3SJZdcos985jPjjrcsa9yEBAAAipdvJ5J+/vOf15e+9CU9+uij6unp0W9+8xs99dRTuu222/zaBAAAKGK+ndMhSSdOnNDf//3fa+/evbIsS0uXLtWaNWt8PWEFAAAUJ19DBwAAwLlw7xUAAGAEoQMAABhB6AAAAEYQOgAAgBGBDB0dHR268847NXfuXM2bN0/r169XugBudOOHzs5O1dbWau/evcPLmpubVV9fr5qaGi1atEjbtm2bxApz8/7772vlypW6/vrrtXDhQq1du1adnZ2SgtGfJO3evVv19fW69tprtXDhQj388MPq7++XFJwepU8uiXzHHXfo/vvvH14WpP5eeeUVXXXVVaqpqRn+19DQICkYfZ48eVJr167VvHnzdN111+nOO+/URx99JCkY/b300ksZz11NTY2uvvpqXX311ZKC0eN7772nFStWaO7cubrhhhv0yCOPDF9yfdL7cwPo61//unvfffe5vb297uHDh90/+7M/c//5n/95ssvK2/79+92vfOUrbnV1tbtnzx7XdV335MmT7vXXX+8+99xz7uDgoPvmm2+6NTU1bnNz8yRXm72+vj534cKF7ve//313YGDA7ezsdL/5zW+63/rWtwLRn+u6bkdHh5tIJNyf/OQnrm3b7ocffuguWbLE/f73vx+YHod873vfc6+88kp33bp1rusG4zV6po0bN7r333//WcuD0ufXv/51d/Xq1W5XV5d76tQp92/+5m/cVatWBaa/kY4fP+4uXLjQ/elPfxqIHm3bdhcuXOj+6Ec/cm3bdo8dO+YuXrzY/cEPflAQ/QXuNx1BvdvtCy+8oDVr1ujee+/NWN7U1KSysjKtWLFCkUhECxYsUF1dXVH1e/ToUV155ZVavXq1otGoysvLtXz5cu3bty8Q/UlSRUWF3nzzTd16660KhUI6efKkBgYGVFFREZgepU9+m9PU1KSvfvWrw8uC1J8ktbS0DP+v+ExB6PPdd99Vc3OzNm7cqNLSUpWUlOjhhx/WmjVrAtHfSK7rqqGhQTfddJNuueWWQPTY1dWljz/+WI7jDN+AzbIsxePxgugvcKFjvLvdFqsbbrhBr732mv70T/80Y3lra6uqq6szllVVVen99983WV5evvCFL+jZZ5/NuIjcq6++qi9+8YuB6G9ISUmJJOnGG29UXV2dKisrdeuttwamx46ODj344IN6/PHHFY/Hh5cHpT9JchxH7733nt544w19+ctf1h/90R/poYceUldXVyD6PHDggKqqqvT888+rtrZWN9xwgx577DFVVlYGor+RXnzxRbW1tQ0fCgxCj+Xl5frGN76hxx57TIlEQjfeeKM+//nP6xvf+EZB9Be40HH69OmMDzxJw9/39vZORkm+qKysHPVmOqP1G4vFirZX13X1xBNP6PXXX9eDDz4YuP6kT/5H/POf/1yWZenuu+8ORI+O46ihoUErV67UlVdemfFYEPob0tnZqauuukqLFy/WK6+8oq1bt+rXv/61GhoaAtFnV1eXPvjgA/3617/WCy+8oJ/+9Kf68MMPtW7dukD0dybHcfT000/r29/+9vB/CILQo+M4isVieuihh/R///d/evnll3Xo0CE9+eSTBdFf4EJHvne7LTbxeHz4ZMQh/f39RdlrT0+P7r77bu3YsUPPPfecZs2aFaj+hsRiMV188cVqaGjQL37xi0D0+MwzzygajeqOO+4467Eg9Dfkoosu0pYtW3TbbbcpHo/r0ksvVUNDg37+85/Ldd2i73PoppsPPvigSkpKdNFFF+mee+7Rz372s0D0d6a9e/fqo48+yrg/WBBeq6+99ppeffVV/cVf/IWi0ahmzpyp1atX68c//nFB9Be40HHm3W6HeLnbbbGprq5Wa2trxrK2tjbNnDlzkirKzeHDh7Vs2TL19PRo+/btmjVrlqTg9PfOO+/oT/7kT4bPIJekVCqlCy64QFVVVUXf44svvqi33npLc+fO1dy5c/Xyyy/r5Zdf1ty5cwPzHEqf/JXVP/7jPw4fK5c+eR4ty9KcOXOKvs+qqio5jqPBwcHhZY7jSJJmz55d9P2d6dVXX1Vtba2mTp06vCwIr9Vjx45lfM5IUiQS0QUXXFAY/Rk7ZdWgr33ta+69997rnjp1avivV5588snJLss3Z/71Smdnpzt37lz3X//1X91UKuXu3r3brampcXfv3j3JVWbv5MmT7k033eTef//9rm3bGY8FoT/Xdd2enh73xhtvdB999FF3YGDAPXLkiHvbbbe53/3udwPT45nWrVs3/NcrQerv2LFj7jXXXOP+8Ic/dAcHB9329nb39ttvdx944IFA9JlKpdza2lr3rrvucnt6etyOjg73L//yL93Vq1cHor8zLVmyxH3++eczlgWhx9bWVvfqq692n376aTedTruHDx92lyxZ4m7cuLEg+gtk6Pj444/du+66y73++uvd+fPnuxs3bnTT6fRkl+WbM0OH67rugQMH3OXLl7s1NTXuH//xH7s/+clPJrE67zZt2uRWV1e7v//7v+9ec801Gf9ct/j7G9La2uquXLnSnTt3rvvlL3/Z/ad/+id3YGDAdd3g9DjkzNDhusHqb+/evcO9zJ8/33344Yfd/v5+13WD0efx48fde+65x124cKE7d+5cd+3atW5XV5frusHob8g111zjvvHGG2ctD0KPu3btcuvr690vfelL7k033VRQnzXcZRYAABgRuHM6AABAYSJ0AAAAIwgdAADACEIHAAAwgtABAACMIHQAAAAjCB0AAMAIQgcAADCC0AEAAIwgdAAAACMIHQAAwAhCBwAAMOL/A+224AI0mPfKAAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "train['Age'].hist(bins=30,color='darkred',alpha=0.3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "id": "059b2e96",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='SibSp', ylabel='count'>"
      ]
     },
     "execution_count": 65,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.countplot(x='SibSp',data=train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "id": "5936f99b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: >"
      ]
     },
     "execution_count": 66,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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z7fOnDljBQIB6yc/+Yn12GOPWRMTE3N5KveFyclJq6ioyPrRj35kXb9+3RobG7OeeeYZ69lnnyW3BDY6Omrl5ORYP/3pT61gMGgNDw9bTz75pPWjH/2I3GzijTfesJYtW2a9+OKLlmXx/8lEd+jQIeull166aft8zo1HQL/mN7/5jU6fPq3a2lq5XC49/PDDeu6559TS0jLXo9332tratHv3br3wwgsR2zs7O5WWlqYtW7YoOTlZhYWFKi4uDmfW2tqqjRs3atWqVUpJSdH27dvldrvV0dExF6dxX7l48aKWLVummpoapaamyu12q7KyUr29veSWwNLT0/Xxxx+rtLRUCxYs0OXLl3X9+nWlp6eTmw10dXWps7NT3/nOd8LbyC2x+Xw+rVix4qbt8zk3CujXDA4OKi0tTQ8++GB425/8yZ/o4sWL+v3vfz+Hk+Hxxx/Xv/zLv+iv/uqvIrYPDg4qMzMzYpvH41F/f78kaWho6I77MXseeeQRHT58OOLLJt5//309+uij5JbgHnjgAUnSunXrVFxcrCVLlqi0tJTcEtzo6Khefvllvfbaa3K5XOHt5Ja4QqGQPvvsM/3bv/2bvv3tb2vt2rV65ZVXND4+Pq9zo4B+zbVr1yL+0UoK/z4xMTEXI+F/LVmyRMnJyTdtv1VmTqcznNfd9sMMy7L0j//4jzp16pRefvllcrOJzs5Offjhh0pKStKuXbvILYGFQiHV1taqqqpKy5Yti9hHbolrbGxM2dnZ2rBhgzo6OnT8+HH9x3/8h2pra+d1bhTQr/nGN76hycnJiG3Tvy9cuHAuRsJduFyu8Jsjpvn9/nBed9uP2Xf16lXt2rVL7e3tOnbsmLKyssjNJpxOpx588EHV1tbql7/8JbklsLfeekupqanaunXrTfvILXEtXrxYLS0tKi8vl8vl0kMPPaTa2lp9+OGHsixr3uZGAf2ajIwMXb58WSMjI+Ftv/71r7V06VItWrRoDifD7WRmZmpwcDBi29DQkDIyMiR9memd9mN2XbhwQWVlZbp69apOnjyprKwsSeSWyD755BP95V/+pQKBQHhbIBBQSkqKPB4PuSWon/3sZzp9+rRWr16t1atX691339W7776r1atX8+8tgfX39+uHP/yhrK98M3ogEFBSUpJyc3Pnb25z/CaohPT0009bL7zwgnXlypXwu+AbGxvneix8xVffBT82NmatXr3a+slPfmIFAgGrq6vLys/Pt7q6uizLssLvGuzq6gq/S/BP//RPrd/97ndzeAb3h8uXL1t/9md/Zr300ktWMBiM2Eduievq1avWunXrrFdffdW6fv269V//9V9WeXm5VVdXR2428uKLL4bfBU9uieu///u/rby8POvtt9+2pqamrC+++ML63ve+Z+3bt29e50YBvYVLly5Zzz//vFVQUGCtWbPGOnTokHXjxo25Hgtf8dUCalmW9emnn1qVlZVWfn6+9ed//ufWT3/604jb//M//7O1YcMGKy8vzyovL7f+/d//3fTI96UjR45YmZmZ1re+9S0rLy8v4seyyC2RDQ4OWlVVVdbq1autb3/729brr79uXb9+3bIscrOLrxZQyyK3RNbT0xPOZs2aNVZ9fb3l9/sty5q/uS2wrK885gsAAADMMl4DCgAAAKMooAAAADCKAgoAAACjKKAAAAAwigIKAAAAoyigAAAAMIoCCgAAAKMooAAAADCKAgoAAACjKKAAAAAwigIKAAAAo/4/A3U4OJdwVz0AAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 800x400 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "train['Fare'].hist(color='green',bins=40,figsize=(8,4))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "9109d9dc",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Data Cleaning"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "1e0accb4",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Want to fill missing age data instead of just dropping the missing age data rows.One way to this is by filling the mean age of all passengers(impulation)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "id": "57af581c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Figure size 1200x700 with 0 Axes>"
      ]
     },
     "execution_count": 67,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<Figure size 1200x700 with 0 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(12,7))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "id": "3664e9ba",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='Pclass', ylabel='Age'>"
      ]
     },
     "execution_count": 68,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAi4AAAGsCAYAAAD62iyRAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8pXeV/AAAACXBIWXMAAA9hAAAPYQGoP6dpAAAvYUlEQVR4nO3dfVRV9YLG8edwEM8RKny70s2W1SA6U4euZSpJy8IY7m1JOSo6hVjO6tqLjVMJ9iLeXgypyCwrncrMJMtJli6Tbmjj1NxrIb5UQM2qgVnrksY1IZVMOIKHPX944ULojePL+e3N+X7WagV7H85+3PzgPOy9z/65LMuyBAAA4AARpgMAAAB0F8UFAAA4BsUFAAA4BsUFAAA4BsUFAAA4BsUFAAA4BsUFAAA4RqTpAGdaa2urjh07poiICLlcLtNxAABAN1iWpdbWVkVGRioi4uTHVXpccTl27JgqKytNxwAAAKfA5/MpKirqpOt7XHFpa2k+n09ut9twGgAA0B2BQECVlZV/82iL1AOLS9vpIbfbTXEBAMBhfu4yDy7OBQAAjkFxAQAAjkFxAQAAjkFxAQAAjkFxAQAAjkFxAQAAjkFxAQAAjkFxAQAAjkFxAQAAjkFxAQAAjmGkuHz55ZfKzMzUyJEjlZycrCeeeELNzc2SpPLycmVkZGjEiBFKSUnRunXrTEQEAAA2FPLi0traqjvuuENpaWnasWOHioqKtG3bNr366qtqaGjQrFmzNHHiRO3cuVN5eXnKz89XRUVFqGMCAAAbCvkkiw0NDaqrq1Nra6ssy5J0fEZnr9erLVu2KDY2VpmZmZKkpKQkpaena82aNUpMTAx1VGMsy5Lf7zeeQfr5ya5CwePx2CIHAMC8kBeXvn376rbbbtNTTz2lp59+WoFAQOPHj9dtt92mJ598UgkJCZ0eHx8fr6KioqC3EwgEzlTkkLIsS3fffbe++OIL01Fsw+fz6aWXXqK8AEAP1t3X7ZAXl9bWVnk8Hi1YsEBTpkxRTU2N7rnnHi1dulRHjhyR1+vt9HiPx6PGxsagt1NZWXmmIoeUZVmn9O/tyY4cOaLPP/+c4gIACH1x+eCDD7R582aVlJRIkoYOHarZs2crLy9P6enpOnz4cKfH+/1+RUdHB70dn88nt9t9RjKH2qpVq4yeKmpqatJNN90kSdq4cWOXMhlqnCoCgJ4vEAh066BDyIvLn//85/Z3ELWHiIxUr169lJCQoI8//rjTuurqag0dOjTo7bjdbscWF0mKiYkxtu2O+y0mJsZ4cQEAoE3I31WUnJysuro6/fu//7sCgYD27Nmj5cuXKz09Xampqaqvr9eqVavU0tKi7du3a9OmTZo8eXKoYwIAABsKeXGJj4/Xyy+/rP/6r//S6NGjNWPGDKWkpOi+++5T3759tXLlSpWUlGj06NHKzc1Vbm6uxowZE+qYAADAhkJ+qkiSrr76al199dUnXOfz+bR27doQJwIAAE7ALf8BAIBjUFwAAIBjUFwAAIBjUFwAAIBjUFwAAIBjUFwAAIBjUFwAAIBjUFwAAIBjUFwAAIBjUFwAAIBjUFwAAIBjUFwAAIBjUFwAAIBjUFwAAIBjUFwAADhF27Zt06RJk7Rt2zbTUcIGxQUAgFPg9/tVUFCgffv2qaCgQH6/33SksEBxAQDgFKxevVr19fWSpPr6ehUWFhpOFB4oLgAABGnv3r0qLCyUZVmSJMuyVFhYqL179xpO1vNRXAAACIJlWVq8ePFJl7eVGZwdFBcAAIJQU1OjsrIyBQKBTssDgYDKyspUU1NjKFl4oLgAABCEIUOGaPTo0XK73Z2Wu91ujR49WkOGDDGULDxQXAAACILL5dLcuXNPutzlchlIFT4oLgAABGnw4MHKyspqLykul0tZWVkaPHiw4WQ9H8UFAIBTMGPGDA0YMECSNHDgQGVlZRlOFB4oLgAAnAKPx6OcnBzFxcUpOztbHo/HdKSwEGk6AAAATpWcnKzk5GTTMcIKR1wAAIBjUFwAAIBjUFwAAIBjUFwAADhF27Zt06RJk7Rt2zbTUcIGxQUAgFPg9/tVUFCgffv2qaCgQH6/33SksEBxAeAY/HULO1m9erXq6+slSfX19SosLDScKDxQXAA4An/dwk727t2rwsLC9pmgLctSYWGh9u7dazhZzxfy4vLuu+9qxIgRnf677LLLdNlll0mSysvLlZGRoREjRiglJUXr1q0LdUQANsRft7ALy7K0ePHiky5vKzM4O0JeXG688UZ99tln7f+VlJQoNjZWeXl5amho0KxZszRx4kTt3LlTeXl5ys/PV0VFRahjArAR/rqFndTU1KisrEyBQKDT8kAgoLKyMtXU1BhKFh6M3jnXsizl5OTo2muv1U033aR169YpNjZWmZmZkqSkpCSlp6drzZo1SkxMDOq5fzqg0H0d910gEGBfwijLsvTMM8+cdPkzzzzDbLwIqcGDB2vUqFHavXt3p9+PbrdbV155pQYPHszvzVPQ3X1mtLhs3LhR1dXVWrZsmSSpqqpKCQkJnR4THx+voqKioJ+7srLyjGQMR0ePHm3/uKKiQr179zaYBuFu37592rFjR5flgUBAO3bs0ObNmxUXF2cgGcLZr3/9a+3ateuEy8vLyw0kCh/Giktra6uWL1+uO++8UzExMZKkI0eOyOv1dnqcx+NRY2Nj0M/v8/nkdrvPSNZw09TU1P5xYmJil+8JEEqWZemDDz446V+3aWlpHHGBETU1NVq9erUsy5LL5dL06dOVmppqOpZjBQKBbh10MFZcysrKtH//fk2ZMqV9mdfr1eHDhzs9zu/3Kzo6Oujnd7vdFJdT1HG/sR9hB9nZ2br55ps7LXO5XMrOzlZkJHPFwoxbb71Vv//971VXV6eBAwfq1ltv5fdlCBh7O/TmzZuVmpqqPn36tC9LSEhQVVVVp8dVV1dr6NChoY4HwEYGDx6srKys9iMrLpdLWVlZGjx4sOFkCGcej0c5OTmKi4tTdna2PB6P6UhhwVhx2b17t6666qpOy1JTU1VfX69Vq1appaVF27dv16ZNmzR58mRDKQHYxYwZMzRgwABJ0sCBA5WVlWU4ESAlJydr/fr1Sk5ONh0lbBgrLnv37tUvfvGLTsv69u2rlStXqqSkRKNHj1Zubq5yc3M1ZswYQykB2AV/3QKQDF7j8tlnn51wuc/n09q1a0OcBoATJCcn85ctEOa45T8AAHAMigsAAHAMigsAAHAMigsAAHAMigsAAHAMigsAx9i2bZsmTZqkbdu2mY4CwBCKCwBH8Pv9Kigo0L59+1RQUCC/3286EgADKC4AHGH16tWqr6+XJNXX16uwsNBwIgAmUFwA2N7evXtVWFgoy7IkHZ8xurCwUHv37jWcDECoUVwA2JplWVq8ePFJl7eVGcAErrsKPYoLAFurqalRWVmZAoFAp+WBQEBlZWWqqakxlAzhjuuuzKC4ALC1IUOGaPTo0XK73Z2Wu91ujR49WkOGDDGUDOGO667MoLgAsDWXy6W5c+eedLnL5TKQCuGO667MobgAsL3BgwcrKyurvaS4XC5lZWVp8ODBhpMhHHHdlVkUFwCOMGPGDA0YMECSNHDgQGVlZRlOhHDFdVdmUVwAOILH41FOTo7i4uKUnZ0tj8djOhLCFNddmUVxAeAYycnJWr9+vZKTk01HQRjjuiuzKC4AAASJ667MobgAAHAKZsyYoXPOOUeSdO6553LdVYhQXAAAOE28kyh0KC4AAJyC1atX6/Dhw5Kkw4cPcwO6EKG4AAAQJG5AZw7FBYBjMKEd7IAb0JlFcQHgCExoB7vgBnRmUVwAOAIT2sEuuAGdWRQXALbH9QSwE25AZxbFBYCtcT0B7Igb0JlDcQFga1xPALuaOnVqp+KSkZFhOFF4oLgAsDWuJ4BdvfPOO2ptbZUktba2at26dYYThQeKCwBb43oC2FHbdVcdcd1VaFBcANhe2/UEHXE9AUzhuiuzKC4AHGHq1KmKiDj+KysiIoLrCWAM112ZZaS4HDp0SPPmzdPo0aN11VVX6e6779b+/fslSeXl5crIyNCIESOUkpLCOUMAko5fT9Dx7dD8boApXHdllpHi8q//+q9qbGzUBx98oA8//FBut1sLFixQQ0ODZs2apYkTJ2rnzp3Ky8tTfn6+KioqTMQEYBPcxwV2wnVXZkWGeoNffPGFysvL9cknnygmJkaStHDhQtXV1WnLli2KjY1VZmamJCkpKUnp6elas2aNEhMTg9rOTw/hofs67rtAIMC+hFGWZemZZ5456fJnnnmGFwqE3Pnnn6/p06dr9erVsixLLpdLmZmZOv/88/mdeYq6u99CXlwqKioUHx+vd955R2+//baampp0zTXX6IEHHlBVVZUSEhI6PT4+Pl5FRUVBb6eysvJMRQ47R48ebf+4oqJCvXv3NpgG4W7fvn3asWNHl+WBQEA7duzQ5s2bFRcXZyAZwl1iYqK8Xq8aGxvVp08fJSYm6vPPPzcdq8cLeXFpaGjQ119/rcsuu0wbNmyQ3+/XvHnz9MADD2jAgAHyer2dHu/xeNTY2Bj0dnw+X5fzj+iepqam9o/bfjABUyzL0gcffKBdu3a13zNDOn49wZVXXqm0tDSOuMAIv9+vyMjjL6Nut1uJiYnyeDyGUzlXIBDo1kGHkBeXqKgoSdL8+fPVu3dvxcTE6N5779XUqVM1adKkLjO++v1+RUdHB70dt9tNcTlFHfcb+xF2kJ2drWnTpnVaZlmWsrOz2184gFBbs2aNDh8+LEk6fPiw3nrrLf32t781nKrnC/nFufHx8WptbVVLS0v7sra/ov7+7/9eVVVVnR5fXV2toUOHhjQjAPvjXhkwiQvGzQl5cbn66qt14YUX6uGHH9aRI0d04MABLVmyRNdff70mTJig+vp6rVq1Si0tLdq+fbs2bdqkyZMnhzomAJtou6nXT08HuVwubvYFI7gBnVkhLy69evVSYWGh3G630tLSlJaWpri4OC1atEh9+/bVypUrVVJSotGjRys3N1e5ubkaM2ZMqGMCsIm2m311vL5FOn6klpt9wQRuQGeWkZPDgwYN0pIlS064zufzae3atSFOBMCu2m72tWvXrk4vFG63WyNHjuRmXwg5xqRZ3PIfgK1xsy/YDWPSLIoLANtjkkXYTduYbCspLpeLMRkiFBcAjnDjjTd2+jw9Pd1QEuC4GTNmaMCAAZKkgQMHdinXODsoLgAc4dFHH+30+WOPPWYmCPAXHo9HOTk5iouLU3Z2NjefCxHu3ATA9nbt2tVlstXy8nLt2rVLI0eONJQKkJKTk5WcnGw6RljhiAsAW2ttbdWCBQtOuG7BggVd3iYNoGejuACwtdLSUjU0NJxwXUNDg0pLS0OcCIBJFBcAtpaUlKTzzjvvhOvOO+88JSUlhTgR8Ffbtm3TpEmTtG3bNtNRwgbFBYCtRUREaOHChSdct3DhQkVE8GsMZvj9fhUUFGjfvn0qKCjoMkkwzg5+4gHY3siRI5WYmNhp2eWXX86FuTBq9erVqq+vlyTV19ersLDQcKLwQHEB4AhPPvlk+9GViIgI5efnG06EcMbs0OZQXAA4QmxsrGbMmKGIiAjNmDFDsbGxpiMhTDE7tFncxwVAt1iWZfwc/vTp0zV9+nS5XC41NTUZy+HxeJiPJoy1zQ79Ux1nh77oootCHyxMUFwA/CzLsnTnnXeqsrLSdBRbSExM1PLlyykvYYrZoc3iVBGAbuFFGjiO2aHN4ogLgJ/lcrm0fPlyo6eKmpqaNGHCBElScXGxvF6vsSycKkLb7NBvvPGGLMtidugQorgA6BaXy2W0LHTk9XptkwXha8aMGXrvvfdUV1fH7NAhxKkiAABOAbNDm8ERFwAAThGzQ4ceR1wAAIBjUFwAAIBjUFwAAIBjUFwAAIBjUFwAAIBjUFwAAIBjUFwAAIBjUFwAAIBjUFwAAIBjUFwAAIBjcMt/AIAjWZZldMbytgySjM8WHk4zllNcAACOY1mW7rzzTlVWVpqOYguJiYlavnx5WJQXThUBABwpHF6k0ZWRIy6///3vlZ2drd69e7cvu/7661VQUKDy8nI98cQTqq6uVt++fXXXXXcpIyPDREwAgE25XC4tX77c6KmipqYmTZgwQZJUXFwsr9drLAunis6yyspK3XTTTcrPz++0vKGhQbNmzdKcOXM0bdo07dy5U7Nnz9awYcOUmJhoIioAwKZcLpfRstCR1+u1TZaezsiposrKSl122WVdlm/ZskWxsbHKzMxUZGSkkpKSlJ6erjVr1hhICQAA7CbkR1xaW1v15Zdfyuv1asWKFQoEAho3bpyys7NVVVWlhISETo+Pj49XUVFR0NsJBAJnKnLY6bjvAoEA+xK2wLiE3TAmz6zu7r+QF5cDBw7oH/7hH5SWlqalS5fq4MGDeuCBB5STk6OBAwd2OdTm8XjU2NgY9Ha40vzUHT16tP3jioqKTtciAaYwLmE3jEkzQl5cBgwY0OnUj9frVU5OjqZOnapJkyZ1udDK7/crOjo66O34fD653e7TzhuOmpqa2j9OTEzkvC1sgXEJu2FMnlmBQKBbBx1CXly++uorFRcXa+7cue1XQDc3NysiIkKJiYl64403Oj2+urpaQ4cODXo7brc76OJih5sZ2UFzc3OnjymAx4XTVft21HEcnsrPN3CmMSbNCHlxiY2N1Zo1a3Teeedp5syZ2r9/vwoKCvRP//RPSktL0+LFi7Vq1SplZmZq9+7d2rRpk5YtWxaSbH6/X+PHjw/Jtpyi7a1+kLZu3cpfVABgWMjfVRQXF6eXX35ZW7du1ahRozR58mT5fD797ne/U9++fbVy5UqVlJRo9OjRys3NVW5ursaMGRPqmAAAwIaM3Mdl1KhRWrt27QnX+Xy+k64LpW/7/k6WK8p0DHP+Mv+GwvzUiMtq1gUHHzcdAwDwF8xVdBKWKyq8i0t49xUAgE0xVxEAAHAMigsAAHAMigsAAHAMigsAAHAMigsAAHAMigsAAHAMigsAAHAMigsAAHAMigsAAHAMigsAAHAMigsAAHAMigsAAHAMigsAAHAMigsAAHAMigsAAHAMigsAAHAMigsAAHAMigsAAHAMigsAAHAMigsAAHAMigsAAHAMigsAAHAMigsAAHCMUy4uBw4cOJM5AAAAflZQxeXYsWNasmSJrrzySqWkpGjPnj2aPHmy9u/ff7byAQAAtAuquLzwwgvavn27nn/+efXq1Uv9+/dXXFyc8vLyzlY+AACAdpHBPHjTpk16++23NWjQILlcLvXp00f5+flKTU09W/kAAADaBXXEpbGxUf369ZMkWZYlSfJ4PIqI4BpfAABw9gXVOH71q1/pxRdflCS5XC5JUmFhoXw+35lPBgAA8BNBnSqaP3++br31Vm3YsEFHjhzRDTfcoCNHjuj1118/W/kAAADaBVVcLrzwQr333nv68MMPVVtbq7i4OF177bWKiYk5W/kAAADaBXWqqLa2VgcPHtSvfvUr3XDDDbriiiv0ww8/qK6uTs3NzUFvPBAIKCsrSw8++GD7svLycmVkZGjEiBFKSUnRunXrgn5eAADQMwV1xCU1NVWtra0nXBcREaGrr75aTz31VPsFvD/nxRdf1K5du3TBBRdIkhoaGjRr1izNmTNH06ZN086dOzV79mwNGzZMiYmJwUQFAAA9UFBHXB566CFdffXVKi4uVnl5ud577z2NGzdOs2fP1oYNGxQTE6P8/PxuPVdpaam2bNmif/zHf2xftmXLFsXGxiozM1ORkZFKSkpSenq61qxZE9y/CgAA9EhBHXF54403tG7dOsXGxkqSLrnkEj311FOaPHmy7rnnHi1cuFDjx4//2ef5/vvvNX/+fC1btkyrVq1qX15VVaWEhIROj42Pj1dRUVEwMSUdPw0Viq9B+AgEAowRgzrue74XsAPG5JnV3f0XVHE5ePCg3G53p2Uul0vff/+9JMnr9Z70VFKb1tZW5eTkaObMmRo+fHindUeOHJHX6+20zOPxqLGxMZiYkqTKysqgv+bo0aNBfw3CR0VFhXr37m06Rtjq+PPJ9wJ2wJg0I6jics0112ju3LmaP3++fvnLX6q2tlZPP/20xo4dq+bmZr300ku69NJL/+ZzvPzyy4qKilJWVlaXdV6vV4cPH+60zO/3Kzo6OpiYkiSfz9elZP2cpqamoLeD8JGYmNilWCN0Ov588r2AHTAmz6xAINCtgw5BFZdHHnlEc+fOVVpaWvsN6K699lrl5eVp165d+uijj/Tss8/+zefYuHGj9u/fr5EjR0o6Xkwk6T//8z81b948ffzxx50eX11draFDhwYTU5LkdruDLi7BPh7h5VTGFM6cjvue7wXsgDFpRlDFJTY2Vq+99pq+++477du3T5Zlaf369UpJSdHnn3+ujRs3/uxzlJSUdPq87a3QTz75pA4ePKiCggKtWrVKmZmZ2r17tzZt2qRly5YFExMAAPRQQRWXNnv27NFrr72m//7v/9bQoUOVk5NzRsL07dtXK1euVF5enpYuXap+/fopNzdXY8aMOSPPDwAAnK3bxaW1tVUlJSV6/fXXVVVVpWPHjunll1/WNddcc1oBnnzyyU6f+3w+rV279rSeEwAA9Ezduo/LG2+8odTUVBUUFCg1NVUfffSRYmJiurx1GQAA4Gzq1hGX/Px83XLLLXrwwQcVFRV1tjMBAACcULeOuCxYsEBlZWUaN26clixZou+++679XUUAAACh0q3ikpmZqffee0/PPvusqqurlZqaqh9++EGlpaXcKRAAAIRMUO8qSkpKUlJSkr799lu99dZbevLJJ/X000/rxhtv7DTDc0/gsoKf7Ro9D+MAAOzllN4OfcEFFygnJ0f/9m//pnfffVdvvfXWmc5lhGVZ7R9fcPBxg0lgRx3HBwDAjKBmh/6pqKgoTZkyRevXrz9TeQAAAE7qlI649FQdLzj+tu/vZLl4B1W4c1nN7UffuCAdAMyjuJyE5YqiuAAAYDMUF8DmLMtqn4w0nHWciZeZ3I/zeDwcCUTYobgANuf3+zV+/HjTMWxlwoQJpiPYwtatW+X1ek3HAELqtC7OBQAACCWOuAAOUpN0uyx3L9MxzGl7S3oYnx5xBVo0pHSF6RiAMRQXwEEsd6/wLi4Awh6nigAAgGNQXAAAgGNQXAAAgGNQXAAAgGNQXAAAgGNQXAAAgGNQXAAAgGNQXAAAgGNQXAAAgGNQXAAAgGNQXAAAgGNQXAAAgGMwySIAICiWZcnv95uOYVxTU9MJPw5nHo9HrrM8ezvFBQAQFL/fr/Hjx5uOYSsTJkwwHcEWtm7dKq/Xe1a3wakiAADgGBxxAQCcsprf3iirVxi/lFjW8f+f5dMjduZqOaYhr74bsu2F8WgDAJwuq1dkeBcXhBynigAAgGMYKS6lpaXKyMjQFVdcobFjx2rhwoXtV6iXl5crIyNDI0aMUEpKitatW2ciIgAAsKGQF5cDBw7ojjvu0M0336xdu3Zpw4YN2rFjh1555RU1NDRo1qxZmjhxonbu3Km8vDzl5+eroqIi1DEBAIANhfzEZL9+/fTJJ58oJiZGlmXp0KFDOnr0qPr166ctW7YoNjZWmZmZkqSkpCSlp6drzZo1SkxMDHVUAABgM0auqIqJiZEkjRs3Tt99951GjhypSZMm6bnnnlNCQkKnx8bHx6uoqCjobQQCgZB8DcJHIBAwMkYYlzgZxiTs5nTGZHe/zuil4Fu2bFFDQ4Oys7M1Z84cDRo0qMuNazwejxobG4N+7srKyqC/5ujRo0F/DcJHRUWFevfuHfLtMi5xMoxJ2E0oxqTR4uLxeOTxeJSTk6OMjAxlZWXp8OHDnR7j9/sVHR0d9HP7fD653e6gvoZbNuNvSUxMPOt3hDwRxiVOhjEJuzmdMRkIBLp10CHkxeXTTz/Vww8/rHfffVdRUVGSpObmZvXq1Uvx8fH6+OOPOz2+urpaQ4cODXo7brc76OIS7OMRXk5lTJ2p7bZxBVpCvn3YS8cxYIcxCXQUijEZ8uIybNgw+f1+LV68WHPnzlVdXZ2eeuopTZkyRWlpaVq8eLFWrVqlzMxM7d69W5s2bdKyZctCHROwDavtzpyShpSuMJgEdtNxbADhIuTFJTo6WitWrNCiRYs0duxYnXPOOUpPT9fs2bMVFRWllStXKi8vT0uXLlW/fv2Um5urMWPGhDomAACwISPXuMTHx2vlypUnXOfz+bR27doQJwLsq+MU8TVJt8ty9zKYBqa5Ai3tR95cYTw/DsIXE0wADmK5e1FcAIQ15ioCAACOQXEBAACOQXEBAACOQXEBAACOwcW5J+Gymk1HMKvt/hBh/q6FsB8HAGAzFJeTuODg46YjAACAn+BUEQAAcAyOuHTg8Xi0detW0zGMa2pq0oQJEyRJxcXFRiZxsyOPx2M6AgCEPYpLBy6Xixfpn/B6vewTAIBtcKoIAAA4BsUFAAA4BsUFAAA4BsUFAAA4BsUFAAA4BsUFAAA4BsUFAAA4BvdxAQCcMlfLMdMRYFioxwDFBQAQFKttElZJQ15912AS2E3HsXG2cKoIAAA4BkdcAABBcblc7R/X/PZGWb14KQlnrpZj7UfeOo6Ns4XRBgA4ZVavSIoLQorRBjiIK9BiOoJZbefPQ/BXnV2F/RhA2KO4AA4ypHSF6QgAYBQX5wIAAMfgiAtgcx6PR1u3bjUdw7impiZNmDBBklRcXCyv12s4kXkej8d0BCDkKC6AzblcLl6kf8Lr9bJPgDDFqSIAAOAYFBcAAOAYFBcAAOAYFBcAAOAYFBcAAOAYFBcAAOAYRorLV199pZkzZ2rUqFEaO3as5s2bpwMHDkiSysvLlZGRoREjRiglJUXr1q0zEREAANhQyIuL3+/X7bffrhEjRmjbtm0qLi7WoUOH9PDDD6uhoUGzZs3SxIkTtXPnTuXl5Sk/P18VFRWhjgkAAGwo5Degq62t1fDhwzV79my53W5FRUVp2rRpmjdvnrZs2aLY2FhlZmZKkpKSkpSenq41a9YoMTExqO0EAoGzET8sdNx3gUCAfQlbYFzaB/seJ3M6P5vd/bqQF5dLLrlEK1Z0nihu8+bNuvTSS1VVVaWEhIRO6+Lj41VUVBT0diorK08rZzg7evRo+8cVFRXq3bu3wTTAcYxL++j4vQA6CsXPptFb/luWpeeee04ffvih3nzzTa1evbrLbbw9Ho8aGxuDfm6fzye3232mooaVpqam9o8TExO5tTpsgXFpHx2/F0BHp/OzGQgEunXQwVhx+fHHH/XQQw/pyy+/1Jtvvqlhw4bJ6/Xq8OHDnR7n9/sVHR0d9PO73W6KyynquN/Yj7ALxqV9sO9xMqH42TTyrqJvvvlGkydP1o8//qiioiINGzZMkpSQkKCqqqpOj62urtbQoUNNxAQAADYT8uLS0NCgW2+9VVdccYVee+019evXr31damqq6uvrtWrVKrW0tGj79u3atGmTJk+eHOqYAADAhkJ+qmj9+vWqra3V+++/r5KSkk7rPvvsM61cuVJ5eXlaunSp+vXrp9zcXI0ZMybUMQEAgA2FvLjMnDlTM2fOPOl6n8+ntWvXhjARAABwCqPvKgIAOJur5ZjpCGZZ1vH/u1xmcxgU6jFAcQEAnLIhr75rOgLCDJMsAgAAx+CICwAgKB6PR1u3bjUdw7impiZNmDBBklRcXMxNEXV8bJxtFBcAQFBcLhcv0j/h9XrZJyHCqSIAAOAYFBcAAOAYFBcAAOAYFBcAAOAYFBcAAOAYFBcAAOAYFBcAAOAYFBcAAOAYFBcAAOAYFBcAAOAYFBcAAOAYFBcAAOAYFBcAAOAYFBcAAOAYFBcAAOAYFBcAAOAYFBcAAOAYFBcAAOAYFBcAAOAYFBcAAOAYFBcAAOAYFBcAAOAYFBcAAOAYFBcAAOAYFBcAAOAYFBcAAOAYFBcAAOAYRovLgQMHlJqaqrKysvZl5eXlysjI0IgRI5SSkqJ169YZTAgAAOzEWHHZvXu3pk2bpm+++aZ9WUNDg2bNmqWJEydq586dysvLU35+vioqKkzFBAAANhJpYqMbNmzQ0qVLlZOTo/vuu699+ZYtWxQbG6vMzExJUlJSktLT07VmzRolJiYGtY1AIHBGM4eTjvsuEAiwL2ELjEvYDWPyzOru/jNSXJKTk5Wenq7IyMhOxaWqqkoJCQmdHhsfH6+ioqKgt1FZWXnaOcPV0aNH2z+uqKhQ7969DaYBjmNcwm4Yk2YYKS4DBw484fIjR47I6/V2WubxeNTY2Bj0Nnw+n9xu9ynlC3dNTU3tHycmJnb5ngAmMC5hN4zJMysQCHTroIOR4nIyXq9Xhw8f7rTM7/crOjo66Odyu90Ul1PUcb+xH2EXjEvYDWPSDFu9HTohIUFVVVWdllVXV2vo0KGGEgEAADuxVXFJTU1VfX29Vq1apZaWFm3fvl2bNm3S5MmTTUcDAAA2YKvi0rdvX61cuVIlJSUaPXq0cnNzlZubqzFjxpiOBgAAbMD4NS5ff/11p899Pp/Wrl1rKA0AALAzWx1xAQAA+FsoLgAAwDEoLgAAwDEoLgAAwDEoLgAAwDEoLgAAwDEoLgAAwDEoLgAAwDEoLgAAwDEoLgAAwDEoLgAAwDEoLgAAwDEoLgAAwDEoLgAAwDEoLgAAwDEoLgAAwDEoLgAAwDEiTQcA4AyWZcnv9xvbflNT0wk/NsHj8cjlchnNAIQriguAn2VZlu68805VVlaajiJJmjBhgtHtJyYmavny5ZQXwABOFQHoFl6kAdgBR1wA/CyXy6Xly5cbPVX0z//8z6qrq2v//Be/+IXefvttI1k4VQSYQ3EB0C0ul0ter9fItt9///1OpUWS9u/fr48++ki/+c1vjGQCYAanigDYWiAQUH5+/gnX5efnKxAIhDgRAJMoLgBsbePGjTp27NgJ1x07dkwbN24McSIAJlFcANjaTTfdpMjIE5/VjoyM1E033RTiRABMorgAsDW3261p06adcN20adPkdrtDnAiASRQXALbW2tqq4uLiE64rLi5Wa2triBMBMIniAsDWSktL1dDQcMJ1DQ0NKi0tDXEiACZRXADYWlJSks4777wTrjvvvPOUlJQU4kQATKK4ALC1iIgITZ8+/YTrpk+frogIfo0B4YQb0NkQk9l1xl1Kw1tra6tef/31E657/fXXdfPNN1NegDBCcbEZJrPrigntwtvHH3+sxsbGE65rbGzUxx9/rGuuuSbEqQCYYsvi8v3332vBggXasWOH3G63brzxRj3wwAMnvZdDT8MLNPBXlmWd1nr0XByd/qtwOjJtyyZw7733atCgQfrjH/+o+vp63XXXXVq1apVuv/1209HOOjtMZif99cXADj8I4fQDia4GDx58WuvRM3F0urNwOjJtu+JSU1OjHTt26A9/+IO8Xq8uvPBC3X333SooKAiL4iKZncwOsJuLL75Yw4cP11dffdVl3fDhw3XxxRcbSAU7CIcXaXRlu+JSVVWl2NhYDRo0qH3Z3/3d36m2tlY//PCDzj333G49DxOvAT3HI488optvvrnL8kcffZQb0IWxF198kaPTf+HxeBz/s9Dd123bFZcjR450OdrQ9nljY2O3i4tdDh8CODOuu+46ffjhh+2fp6SkqK6uTnV1dQZTAQg12xWXPn36dLnIqe3z6Ojobj+Pz+djDhOgBxk+fLi2b9+upqYm9enTRw899JA8Ho/pWADOkEAg0K2DDrYrLkOHDtWhQ4dUX1+vAQMGSJL+7//+T3FxcTrnnHO6/Txut5viAvQg0dHReuyxx/Tss8/q/vvvD+oPGQA9h+3u2nTRRRfpyiuv1KJFi/Tjjz9qz549WrZsmaZMmWI6GgDDkpOTtX79eiUnJ5uOAsAQ2xUXSVq6dKmOHTum8ePHa+rUqbrmmmt09913m44FAAAMs92pIkkaMGCAli5dajoGAACwGVsecQEAADgRigsAAHAMigsAAHAMigsAAHAMigsAAHAMigsAAHAMigsAAHAMigsAAHAMW96A7nS0TTHe3emxAQCAeW2v222v4yfT44pLa2urJHVrhkkAAGAvba/jJ+Oyfq7aOExra6uOHTumiIgIuVwu03EAAEA3WJal1tZWRUZGKiLi5Fey9LjiAgAAei4uzgUAAI5BcQEAAI5BcQEAAI5BcQEAAI5BcQEAAI5BcQEAAI5BcQEAAI5BccFJHThwQKmpqSorKzMdBWHuq6++0syZMzVq1CiNHTtW8+bN04EDB0zHQpgrLS1VRkaGrrjiCo0dO1YLFy6U3+83HavHo7jghHbv3q1p06bpm2++MR0FYc7v9+v222/XiBEjtG3bNhUXF+vQoUN6+OGHTUdDGDtw4IDuuOMO3Xzzzdq1a5c2bNigHTt26JVXXjEdrcejuKCLDRs2KDs7W/fdd5/pKIBqa2s1fPhwzZ49W1FRUerbt6+mTZumnTt3mo6GMNavXz998sknmjRpklwulw4dOqSjR4+qX79+pqP1eBQXdJGcnKwPPvhAN9xwg+kogC655BKtWLFCbre7fdnmzZt16aWXGkwFSDExMZKkcePGKT09XQMHDtSkSZMMp+r5KC7oYuDAgYqM7HETh6MHsCxLS5Ys0Ycffqj58+ebjgNIkrZs2aI//OEPioiI0Jw5c0zH6fEoLgAc4ccff9ScOXO0adMmvfnmmxo2bJjpSIAkyePxaNCgQcrJydEf//hHNTQ0mI7Uo1FcANjeN998o8mTJ+vHH39UUVERpQXGffrpp/r1r3+t5ubm9mXNzc3q1auXvF6vwWQ9H8UFgK01NDTo1ltv1RVXXKHXXnuNix9hC8OGDZPf79fixYvV3Nysb7/9Vk899ZSmTJmiqKgo0/F6NC5kAGBr69evV21trd5//32VlJR0WvfZZ58ZSoVwFx0drRUrVmjRokUaO3aszjnnHKWnp2v27Nmmo/V4LsuyLNMhAAAAuoNTRQAAwDEoLgAAwDEoLgAAwDEoLgAAwDEoLgAAwDEoLgAAwDEoLgAAwDEoLgAAwDG4cy6AsyolJUV1dXXtM45blqWYmBilp6crJydHEREn//spJSVF99xzjyZNmhSquABsjuIC4Kx77LHHOpWPr7/+Wrfddpu8Xq/mzJljMBkAp+FUEYCQGzZsmK666ir9z//8jxobG/X4448rKSlJI0eO1G9/+1t9++23Xb7mu+++07333quUlBRdfvnlGj9+vIqKitrXv/XWW7r++us1cuRIpaena926de3rXnjhBY0bN06jRo3S5MmTtXXr1pD8OwGceRQXACHV0tKisrIybd++XWPHjtXjjz+uyspKrV+/Xp988okGDBig+++/v8vX5ebmqlevXnrvvff06aefavr06Vq4cKGOHDmiPXv2KD8/X6+88op27dqlefPmaeHChdq/f7+2b9+u//iP/9C6detUVlamjIwMzZ8/Xy0tLQb+9QBOF6eKAJx1jz32mBYtWtT+eVxcnGbOnKlp06bpyiuv1PLly3X++edLkh566CHV1NR0eY4nnnhC0dHR6tWrl2praxUdHS2/36+Ghga53W5ZlqW1a9cqLS1NSUlJ+vzzzxUREaFvv/1WDQ0Neuedd3TdddcpIyND06ZNk8vlCtm/H8CZQ3EBcNY98sgjJ7zAtq6uTs3NzfrlL3/Zvuzcc8+Vz+fr8tg9e/bo6aef1p/+9CdddNFFGjJkiCSptbVVgwcPVmFhoVasWKE777xTgUBAkyZNUk5OjkaMGKEXXnihfb3H41FWVpbuuuuuv3lhMAB7orgAMKZ///6KiorSn//8Z11yySWSpO+//16vvvqq7r333vbHtbS06I477tD999+vW265RS6XS1988YXefffd9q8JBAJ66aWX1Nraqk8//VRz5szRxRdfrOuuu079+/fXa6+9pubmZpWWluqee+7RpZdeqmuvvdbAvxrA6eDPDQDGREREaOLEiXrhhRf03Xff6ejRo3ruuef0+eefy+PxtD+upaVFfr9fHo9HLpdLtbW1KigoaF9XW1urf/mXf1FpaakiIiI0aNAgSVLfvn1VWVmp22+/XV999ZWioqLUv3//9nUAnIcjLgCMevDBB7VkyRJlZGTI7/dr1KhRev755zs9pk+fPlq0aJGef/55PfHEE+rfv7+mTp2q6upq/e///q/S0tL0u9/9To8++qj279+vc845R7fccot+85vfyOVy6U9/+pPuuusuHTx4UP3799fDDz+syy+/3NC/GMDpcFmWZZkOAQAA0B2cKgIAAI5BcQEAAI5BcQEAAI5BcQEAAI5BcQEAAI5BcQEAAI5BcQEAAI5BcQEAAI5BcQEAAI5BcQEAAI5BcQEAAI7x/8e6hHsMAXy0AAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.boxplot(x='Pclass', y='Age',data =train,palette ='winter')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "c735c26c",
   "metadata": {},
   "outputs": [],
   "source": [
    "# we can see the wealthier passenegers in the higher classes tend to be older. \n",
    "#will use these averange age values to impulate based on Pclass for Age."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "id": "32432c48",
   "metadata": {},
   "outputs": [],
   "source": [
    "def impute_age(cols):\n",
    "    Age = cols[0]\n",
    "    Pclass = cols[1]\n",
    "    if pd.isnull(Age):\n",
    "        if Pclass == 1:\n",
    "            return\n",
    "        elif Pclass == 2:\n",
    "            return 29\n",
    "        else:\n",
    "            return 24\n",
    "    else:\n",
    "        return Age"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "id": "c59a4c5f",
   "metadata": {},
   "outputs": [],
   "source": [
    "train['Age'] =train[['Age','Pclass']].apply(impute_age,axis =1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "id": "c01c813e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: >"
      ]
     },
     "execution_count": 73,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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nj1ba9asUVpamulI/3P2799/zeubN2/2chJ4EwcVFWDXrl2Ffk/dunW9kOSPy87O1k033SRJ2rRpk8qUKaPq1asbTpW/06dP6+WXX9bBgwcVEBCg5cuXKzw8XPPnz9fDDz9sOt411ahRQz169NDf/vY3+fn5KSkpSYMHD9b333+vrVu3mo53lSlTpmj79u169dVXFRERoU2bNmnw4MHy9/fX1KlTTcfL1549e/Tiiy+qRIkSqlChghITE+U4jhYsWKAqVaqYjvc/o1atWtq7d2+ea2lpaQoNDdXXX39tKBVuNK+WgapVq8rPz6/A7zl8+LCX0hSuatWqkpQnc6lSpXTu3Dnl5OSodOnS2rZtm6l4hdqwYYOGDRumr776SjNmzNDMmTPl5+enyMhIderUyXS8axo4cKBKliypN998U40bN9auXbv07rvvavPmzVq6dKnpeNf0zTffKCIiQrfffrs6dOig8ePHq169eho5cqTKli1rOt5VmjVrpqVLl+r2229XvXr1tHPnTqWmpqpFixbasWOH6Xj56tixo1q0aKF+/fpJkhzH0bRp07Rz504tXLjQcLr87d+/X/Hx8crOzs5zvX379mYCXcPPP/+sxx9/XNnZ2XIc55rv07Vq1dLixYsNpLs+W7du1cKFC3Xy5EnNmjVL8+fP18CBA+Xv72862lXc+EHTq/9KH3zwgaTL/2mbN2/WK6+8orvuukvHjx/X9OnTFRIS4s04hTpy5Igkad68efr22281bNgw3XLLLTp//rzGjx+vUqVKGU5YsHfffVcDBgxQTk6OFi1apJiYGAUGBioiIsK1ZWD79u367LPPVKJECc8bUq9evTR//nzDyfJXrVo1ffTRR2rfvr2GDh2qp556SqNGjTIdK1/nz5/3lJTczwLFixdXkSLuHjX84Ycf1KtXL89rPz8/9evXT++99565UIWYMmWKZs+erXLlyikgIMBz3c/Pz1Vl4O6779ZHH32k1NRU9enTR3PmzMnz9WLFiukvf/mLoXSFi4uLU1RUlJ566inPjXbDhg3y8/PT66+/bjjd1bp16ybJZR80HQMeeeQR58SJE3munTx50gkLCzMRp1ANGjRwLly4kOfaxYsXnXr16hlKdH1y8x06dMipWbOmk5mZ6TiO49SsWdNkrAKFhYU5KSkpjuM4Tp06dRzHcZwzZ8649mfDcRzn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yQoKClKHDh0883fcaurUqZo/f75Kly6tkSNHqkmTJqYjFWjFihUaPXq0Ll26lOd6YGBgvkeMu4Gb9kcwUgYK2iUsd2zbTdavX6/mzZu78vSrgvjiL8iECRMUFBSk5557znSUQg0ZMkQZGRmKjIxUYGCgWrZsqUcffVQRERGKjY3VmjVrNHv2bNMx88jJydG//vUvLViwQN99952ysrL07rvvXjWu7Qtyx7bd7MUXX9SkSZNcXVZyXbkZ1TvvvKOjR4/qnXfe8bz3uXV4o0WLFuratatKliypXbt26fnnn9ekSZMUEhKi3r17m45XILd80DS66dDZs2d17NgxVatWTVlZWa7d+axx48bKzMxU+/btFR4e7vo3oFy++AvStWtX7dmzRyVLllSpUqXyPD26ct6GaaGhoVq9erXKli2rxMRENWvWTGvXrlXlypWVnp6upk2bump73Pfff18ffPCBcnJy1LlzZ3Xq1EmtW7fW6tWrdfvtt5uOd13S0tK0adMmJSUl6c4771Tjxo1VvHhx07Hy9de//lWbN2927Xvbb115xHzurcHPz89zrLFbn9zWrFlTX3/9tRISEjRo0CD985//VGJiorp3765PPvnEdLx8JScnX3NC97Jly7x+sqyRj7rp6ekaMWKE1q5dq+LFi2vlypXq0aOHFixY4Mr90Tdu3Kgvv/xSq1at0pNPPqkHHnhA4eHheuyxx1y9IiI5OVnPP/+8EhIStGLFCgUHB2vcuHHq3r27a8tAeHi4wsPDTce4LmlpaZ5d+vbv369bb73VUxSLFSumzMxMk/GuEhUVpS5dumjIkCE+cXO60oEDB9SrVy8VL15cFSpUUEJCgooWLaq5c+e68n1D8q0lb24r279HYGCgMjMzFRQUpB9//FHS5acYV+777zY9e/bUokWLVKpUKUmXd4wdOnSodu/ebUcZmDhxos6fP6//+7//U6dOnVSpUiU1bdpUY8eO1bx580xEKlCRIkUUFhamsLAwnTt3TuvWrdOMGTM0btw47d2713S8fPniL0iHDh2ueT0rK8vLSQpXqlQppaSkqGzZstq5c2eeceAffvjBdadDDh8+XEuWLFFYWJg6deqkLl26uP5sjd+KiopSjx491K9fP0mXP7lGR0dr1KhReu+998yGy4cvLXm74447JF2eVzRt2jSFh4erUqVKev/993XmzBmv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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "id": "4f17972d",
   "metadata": {},
   "outputs": [],
   "source": [
    "train.drop('Cabin',axis=1,inplace =True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "id": "6e0b8f9b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Braund, Mr. Owen Harris</td>\n",
       "      <td>male</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>A/5 21171</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>\n",
       "      <td>female</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>PC 17599</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>Heikkinen, Miss. Laina</td>\n",
       "      <td>female</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>STON/O2. 3101282</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>\n",
       "      <td>female</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>113803</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Allen, Mr. William Henry</td>\n",
       "      <td>male</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>373450</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   PassengerId  Survived  Pclass  \\\n",
       "0            1         0       3   \n",
       "1            2         1       1   \n",
       "2            3         1       3   \n",
       "3            4         1       1   \n",
       "4            5         0       3   \n",
       "\n",
       "                                                Name     Sex   Age  SibSp  \\\n",
       "0                            Braund, Mr. Owen Harris    male  22.0      1   \n",
       "1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   \n",
       "2                             Heikkinen, Miss. Laina  female  26.0      0   \n",
       "3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   \n",
       "4                           Allen, Mr. William Henry    male  35.0      0   \n",
       "\n",
       "   Parch            Ticket     Fare Embarked  \n",
       "0      0         A/5 21171   7.2500        S  \n",
       "1      0          PC 17599  71.2833        C  \n",
       "2      0  STON/O2. 3101282   7.9250        S  \n",
       "3      0            113803  53.1000        S  \n",
       "4      0            373450   8.0500        S  "
      ]
     },
     "execution_count": 72,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9456fd66",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
