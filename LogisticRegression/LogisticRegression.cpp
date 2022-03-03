#include <iostream>
#include <vector>
#include <fstream>
#include "bits-stdc++.h"

using namespace std;

string DoubletoString(double a)
{
    ostringstream temp;
    temp << a;
    return temp.str();
}

template<typename T>
void drop_rows(vector<vector<T>>& tdata, const int& i)
{
    tdata[i].clear();
}


void featurize_data(vector<vector<string>>& tdata, vector<int>& alphafeature)
{
    for (int j = 0; j < alphafeature.size(); j++)
    {
        vector<string> data;
        for (int i = 0; i < tdata.size(); i++)
        {
            data.push_back(tdata[i][alphafeature[j]]);
        }
        sort(data.begin(), data.end());
        data.erase(unique(data.begin(), data.end()), data.end());

        for (int c = 0; c < data.size(); c++)
        {
            for (int i = 0; i < tdata.size(); i++)
            {
                if (tdata[i][alphafeature[j]] == data[c])
                {
                    tdata[i][alphafeature[j]] = DoubletoString(c);
                }
            }
        }

    }

}


void drop_Nan_row(vector<vector<string>>& tdata)
{
    for (int i = 0; i < tdata.size(); i++)
    {
        for (int j = 0; j < tdata[i].size(); j++)
        {
            if (tdata[i][j] == "Nan" || tdata[i][j] == "nan" || tdata[i][j] == "NAN")
            {
                tdata[i].clear();

            }
        }
    }
}


template<typename T>
void reposition_target_col(vector<vector<T>>& tdata, const int& current)
{
    for (int i = 0; i < tdata.size(); i++)
    {
        string td = tdata[i][current];
        tdata[i].erase(tdata[i].begin() + current);
        tdata[i].push_back(td);
    }

}

vector<string> split(const string& str, const string& delim)
{
    vector<string> tokens;
    size_t prev = 0, pos = 0;
    do
    {
        pos = str.find(delim, prev);
        if (pos == string::npos)
        {
            pos = str.length();
        }
        string token = str.substr(prev, pos - prev);
        if (!token.empty())
        {
            tokens.push_back(token);
        }
        prev = pos + delim.length();
    } while (pos < str.length() && prev < str.length());

    return tokens;
}


template<typename T>
void column_drop(vector<int> drops, vector<vector<T>>& tdata)
{
    sort(drops.begin(), drops.end());
    for (int k = 0; k < drops.size(); k++)
    {
        if (k > 0)
        {
            drops[k] = drops[k] - k;
        }
        for (int i = 0; i < tdata.size(); i++)
        {
            tdata[i].erase(tdata[i].begin() + drops[k]);
        }

    }
}



template<typename T>
void shuffle_rows(vector<vector<T>>& tdata, time_t seed)
{
    srand((unsigned)seed);
    vector<T> saved;
    for (int i = 1; i < tdata.size(); i++)
    {
        int r = rand() % tdata.size();
        if (r != i && r != 0)
        {
            for (int j = 0; j < tdata[i].size(); j++)
            {
                saved.push_back(tdata[i][j]);
            }
            drop_rows(tdata, i);
            for (int j = 0; j < saved.size(); j++)
            {
                tdata[i].push_back(tdata[r][j]);
            }
            drop_rows(tdata, r);
            for (int j = 0; j < saved.size(); j++)
            {
                tdata[r].push_back(saved[j]);
            }
            saved.clear();

        }

    }
}



vector<double> initialize_para(vector<vector<double>>& tdata)
{
    vector<double> slopes;
    set<double> order;
    double Ymin = 0, Xmin = 0, Ymax = 0, Xmax = 0, Ydif = 0;

    for (int i = 0; i < tdata.size(); i++)
    {
        order.insert(tdata[i][tdata[i].size() - 1]);
    }

    Ymin = *next(order.begin(), 0);
    Ymax = *next(order.begin(), order.size() - 1);
    Ydif = Ymax - Ymin;
    order.clear();

    for (int j = 0; j < tdata[0].size() - 1; j++)
    {
        for (int i = 0; i < tdata.size(); i++)
        {
            order.insert(tdata[i][j]);
        }
        Xmin = *next(order.begin(), 0);
        Xmax = *next(order.begin(), order.size() - 1);
        slopes.push_back(Ydif / (Xmax - Xmin));
        order.clear();
    }

    return slopes;

}

double predict_Y(const int i, vector<vector<double>>& tdata, vector<double>& slopes, double& intercept)
{
    double Z = intercept;
    for (int j = 0; j < tdata[i].size() - 1; j++)
    {
        Z += (slopes[j] * tdata[i][j]);
    }

    return (1 / (1 + exp(-Z)));
}

double Pdifferentiate_loss_wst_intercept(vector<vector<double>>& tdata, vector<double>& predictions)
{
    double dif = 0;
    for (int i = 0; i < tdata.size(); i++)
    {
        dif += (predictions[i] - tdata[i][tdata[i].size() - 1]);
    }

    return dif;
}

double Pdifferentiate_loss_wst_slope(const int j, vector<vector<double>>& tdata, vector<double>& predictions)
{
    double dif = 0;
    for (int i = 0; i < tdata.size(); i++)
    {
        dif += (tdata[i][j] * (predictions[i] - tdata[i][tdata[i].size() - 1]));
    }

    return dif;
}

void Gradient_Descend(vector<vector<double>>& tdata, vector<double>& predictions, vector<double>& slopes, double& intercept, const double learning_rate)
{
    intercept = intercept - (learning_rate * Pdifferentiate_loss_wst_intercept(tdata, predictions));

    for (int j = 0; j < slopes.size(); j++)
    {
        slopes[j] = slopes[j] - (learning_rate * Pdifferentiate_loss_wst_slope(j, tdata, predictions));
    }

}

double Bcross_entropy(vector<vector<double>>& tdata, vector<double>& slopes, double& intercept, vector<double>& predictions)
{
    double T = 0;
    for (int i = 0; i < tdata.size(); i++)
    {
        double predicted = predict_Y(i, tdata, slopes, intercept);
        predictions[i] = predicted;
        T += ((tdata[i][tdata[i].size() - 1] * log(predicted)) + ((1 - tdata[i][tdata[i].size() - 1]) * log((1 - predicted))));

    }

    return -(T / (tdata.size() - 1));
}

template<typename T>
void cal_Accuracy(vector<T>& prediction, vector<T> Actuals)
{
    if (prediction.size() != Actuals.size())
    {
        cout << "something wrong with data-set-size!" << endl;
    }
    else
    {
        double correct = 0;
        for (int i = 0; i < prediction.size(); i++)
        {
            if (prediction[i] == Actuals[i])
            {
                correct++;
            }
        }

        cout << "Accuracy Score: " << ((correct / prediction.size()) * 100) << " %" << endl;
    }

}

void Logistic_Regression(vector<vector<double>>& tdata, const double learning_rate, const double steps, vector<double>& slopes, double& intercept, const double split_train_per)
{
    vector<vector<double>> testdatasplited;
    vector<double> actuals;
    const int testdataN = (tdata.size()) - ((split_train_per * (tdata.size())) / 100);
    testdatasplited.resize(testdataN);
    int y = 0;

    for (int i = tdata.size() - testdataN; i < tdata.size(); i++)
    {
        actuals.push_back(tdata[i][tdata[i].size() - 1]);
        for (int j = 0; j < tdata[i].size(); j++)
        {
            testdatasplited[y].push_back(tdata[i][j]);
        }
        y++;

    }

    tdata.resize(tdata.size() - testdataN);
    tdata.shrink_to_fit();

    double loss = 0;
    vector<double> predictions;

    predictions.resize(tdata.size());
    slopes = initialize_para(tdata);

    cout << "loss--> " << Bcross_entropy(tdata, slopes, intercept, predictions) << endl;
    int i = 0;
    while (i < steps)
    {
        Gradient_Descend(tdata, predictions, slopes, intercept, learning_rate);
        loss = Bcross_entropy(tdata, slopes, intercept, predictions);
        cout << "loss--> " << loss << endl;

        i++;

    }

    vector<double> testpredictions;

    for (int i = 0; i < testdatasplited.size(); i++)
    {
        double p = predict_Y(i, testdatasplited, slopes, intercept);
        if (p >= 0.5)
        {
            p = 1;
        }
        else
        {
            p = 0;
        }

        testpredictions.push_back(p);
    }

    cal_Accuracy(testpredictions, actuals);

}

vector<vector<double>> readprepareTraindataset(const char* fname)
{
    vector<string> data;
    vector<vector<string>> tdata;
    ifstream file(fname);
    string line = "";
    int u = 0;
    while (getline(file, line))
    {
        if (line != "")
        {
            for (int i = 0; i + 1 < line.length(); i++)
            {
                if (line[i] == ',' && line[i + 1] == ',')
                {
                    line.insert(i + 1, "0");
                }
            }
            data.push_back(line);
            u++;
        }

    }

    file.close();

    tdata.resize(data.size() - 1);

    for (int i = 1; i < data.size(); i++)
    {
        vector<string> str = split(data[i], ",");

        for (int j = 0; j < str.size(); j++)
        {
            tdata[i - 1].push_back(str[j]);

        }
    }

    reposition_target_col(tdata, 0);
    vector<int> alpha = { 2 };
    featurize_data(tdata, alpha);
    shuffle_rows(tdata, 4);

    vector<vector<double>> tdataN;

    tdataN.resize(tdata.size());

    for (int i = 0; i < tdata.size(); i++)
    {
        for (int j = 0; j < tdata[i].size(); j++)
        {
            tdataN[i].push_back(atof(tdata[i][j].c_str()));
        }

    }

    return tdataN;

}

int main()
{
    vector<pair<int, double>> rel;
    vector<double> slopes;
    double intercept = 0;
    vector<vector<double>> tdata = readprepareTraindataset("01_heights_weights_genders.csv");

    Logistic_Regression(tdata, 0.00000005, 9000, slopes, intercept, 70);


    return 0;
}
