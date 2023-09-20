#include <iostream>
#include <vector>
#include <cmath>
#include <utility> 
#include <fstream>

# define M_E 2.7182818284590452354 /* e */

class DNN
{
    public:
    std::pair<double, double> output_linear_regression;
    std::pair<double, double> output_logistic_regression;
    double cost;
    double logistic_cost;

    DNN(const std::vector<std::vector<double>> &x_batch_, const std::vector<int> &y_, const int &linear_epoch_, const int &logistic_epoch_,
         const float &linear_lr_, const float &logistic_lr_)
    {
        x_batch = x_batch_;
        y = y_;

        linear_epoch = linear_epoch_;
        logistic_epoch = logistic_epoch_;

        linear_lr = linear_lr_;
        logistic_lr = logistic_lr_;

        output_linear_regression = DNN::learning_function(x_batch,y);
        output_logistic_regression = DNN::logistic(w,b);
    }


    private:
    std::vector<std::vector<double>> x_batch;
    std::vector<int> y;

    double w;
    double b;

    double sigma_w_gradient;
    double sigma_b_gradient;

    int count_y;
    int count_x;

    int linear_epoch;
    int logistic_epoch;

    float linear_lr;
    float logistic_lr;

    std::pair<double, double> learning_function(const std::vector<std::vector<double>> &x, const std::vector<int> &y)
    {
        for(int i = 0; i < linear_epoch; i++)
        {

            sigma_w_gradient += (2 * std::pow(x[count_y][count_x],2) * w) + (2 * x[count_y][count_x] * b) - (2 * x[count_y][count_x] * y[count_y]);
            sigma_b_gradient += (2 * x[count_y][count_x] * w) - (2 * y[count_y]) + (2 * b);


            if(count_y == (x.size() - 1) && count_x == (x[0].size() - 1))
            {
                sigma_w_gradient /= x.size() * x[0].size();
                sigma_b_gradient /= x.size() * x[0].size();

                w = w - (linear_lr * sigma_w_gradient);
                b = b - (linear_lr * sigma_b_gradient);

                sigma_w_gradient = 0.0;
                sigma_b_gradient = 0.0;
            }

            // if((i % 500) == 0)
            // {
            //     std::cout << "w : " << w << "           " << "b : " << b << "           "  << "cost :" << (w * x[count][count_] + b) - y[count] <<std::endl;
            // }

            count_x++;

            if(count_x == x[0].size())
            {
                count_x = 0;
                count_y++;
                if(count_y == x.size())
                {
                    count_y = 0;
                }
            }
            cost = (w * x[count_y][count_x] + b) - y[count_y];
        }

        return std::make_pair(w,b);
    }

    std::pair<double, double> logistic(double w, double b)
    {
        int count_y = 0;
        int count_x = 0;
        
        for(int i = 0; i < logistic_epoch; i++)
        {
            sigma_w_gradient += -((y[count_y]- 1 / (1+pow(M_E, -(w * x_batch[count_y][count_x] + b)))) * x_batch[count_y][count_x]);
            sigma_b_gradient += -(y[count_y]- 1 / (1+pow(M_E, -(w * x_batch[count_y][count_x] + b))));
 

            if(count_y == (x_batch.size() - 1) && count_x == (x_batch[0].size() - 1))
            {
                sigma_w_gradient /= x_batch.size() * x_batch[0].size();
                sigma_b_gradient /= x_batch.size() * x_batch[0].size();

                w -= (logistic_lr * sigma_w_gradient);
                b -= (logistic_lr * sigma_b_gradient);

                sigma_w_gradient = 0.0;
                sigma_b_gradient = 0.0;
            } 

            if((i % 1000) == 0)
            {
                std::cout << "w : " << w << "           " << "b : " << b << "           "  << "cost :" << logistic_cost << std::endl << std::endl;
            }

            count_x ++;

            if(count_x == x_batch[0].size())
            {
                count_x = 0;
                count_y++;
                if(count_y == x_batch.size())
                {
                    count_y = 0;
                }
            }

            logistic_cost = y[count_y] -1 / (1+pow(M_E, -(w * x_batch[count_y][count_x] + b)));
        }

        return std::make_pair(w,b);
    }
};


int main()
{
    int row = 2;
    int column = 10; 

    double count = 1.0;

    int linear_epoch = 200000;
    int logistic_epoch = 10000000;

    float linear_lr = 0.01;
    float logistic_lr = 0.01;

    std::vector<std::vector<double>> x_batch(row,std::vector<double>(column,0));

    for(int i = 0; i < row; i++)
    {
        for(int z = 0; z < column; z++)
        {
            x_batch[i][z] = count;

            count += 0.5;
        }
    }

    std::ofstream outputFile("data/logistic_regression_data.txt");

    for (int i = 0; i < row; i++) 
    {
        for (int z = 0; z < column; z++) {
            outputFile << "(" << x_batch[i][z] << "," << i << ")" << "\n";
        }
        outputFile << "\n"; 
    }
    

    DNN data_obj(x_batch, {0,1}, linear_epoch, logistic_epoch, linear_lr, logistic_lr);

    std::cout << "Linear Regression  : " << std::endl;
    std::cout << "Equation: y = " << data_obj.output_linear_regression.first << "x + " << data_obj.output_linear_regression.second <<std::endl;
    std::cout << "Final cost : " << data_obj.cost << std::endl << std::endl;

    outputFile << "\n" << "linear_regression : y = " << data_obj.output_linear_regression.first << "x + " << data_obj.output_linear_regression.second;

    std::cout << "Logistic Regression  : " << std::endl;
    std::cout << "Equation: y = " << data_obj.output_logistic_regression.first << "x + " << data_obj.output_logistic_regression.second <<std::endl;
    std::cout << "Final cost : " << data_obj.logistic_cost << std::endl << std::endl;

    outputFile << "\n" << "logistic_regression : y = " << data_obj.output_logistic_regression.first << "x + " << data_obj.output_logistic_regression.second << "\n";
    outputFile << "for Desmos : 1/(1+e^{-({" << data_obj.output_logistic_regression.first << "x + " << data_obj.output_logistic_regression.second << "})})";

    return 0;
}