{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import pandas as pd\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "dataset = pd.read_csv('E:\\Machine Learning A-Z (Codes and Datasets)\\Part 2 - Regression\\Section 4 - Simple Linear Regression\\Python\\Salary_Data.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = dataset.iloc[:, :-1].values\n",
    "y = dataset.iloc[:, -1].values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
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
       "      <th>YearsExperience</th>\n",
       "      <th>Salary</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1.1</td>\n",
       "      <td>39343.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1.3</td>\n",
       "      <td>46205.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1.5</td>\n",
       "      <td>37731.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>2.0</td>\n",
       "      <td>43525.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>2.2</td>\n",
       "      <td>39891.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>2.9</td>\n",
       "      <td>56642.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>3.0</td>\n",
       "      <td>60150.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>3.2</td>\n",
       "      <td>54445.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>3.2</td>\n",
       "      <td>64445.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>3.7</td>\n",
       "      <td>57189.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10</th>\n",
       "      <td>3.9</td>\n",
       "      <td>63218.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11</th>\n",
       "      <td>4.0</td>\n",
       "      <td>55794.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>12</th>\n",
       "      <td>4.0</td>\n",
       "      <td>56957.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>13</th>\n",
       "      <td>4.1</td>\n",
       "      <td>57081.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>14</th>\n",
       "      <td>4.5</td>\n",
       "      <td>61111.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    YearsExperience   Salary\n",
       "0               1.1  39343.0\n",
       "1               1.3  46205.0\n",
       "2               1.5  37731.0\n",
       "3               2.0  43525.0\n",
       "4               2.2  39891.0\n",
       "5               2.9  56642.0\n",
       "6               3.0  60150.0\n",
       "7               3.2  54445.0\n",
       "8               3.2  64445.0\n",
       "9               3.7  57189.0\n",
       "10              3.9  63218.0\n",
       "11              4.0  55794.0\n",
       "12              4.0  56957.0\n",
       "13              4.1  57081.0\n",
       "14              4.5  61111.0"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dataset.head(15)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "LinearRegression()"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.linear_model import LinearRegression\n",
    "regressor = LinearRegression()\n",
    "regressor.fit(X_train, y_train)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "y_pred = regressor.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0, 0.5, 'Salary')"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.scatter(X_train, y_train, color='green')\n",
    "plt.plot(X_train, regressor.predict(X_train))\n",
    "plt.title('Salary vs Experience')\n",
    "plt.xlabel('Years of Exp')\n",
    "plt.ylabel('Salary')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0, 0.5, 'Salary')"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.scatter(X_test, y_test, color='green')\n",
    "plt.plot(X_train, regressor.predict(X_train), color='red')\n",
    "plt.title('Salary vs Experience')\n",
    "plt.xlabel('Years of Exp')\n",
    "plt.ylabel('Salary')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
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
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
