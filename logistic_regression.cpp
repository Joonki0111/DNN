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

    DNN(const std::vector<std::vector<double>> &x_batch_, const std::vector<int> &y_)
    {
        x_batch = x_batch_;
        y = y_;
        output_linear_regression = DNN::learning_function(x_batch,y);
        output_logistic_regression = DNN::logistic(w,b);
    }


    private:
    std::vector<std::vector<double>> x_batch;
    std::vector<int> y;
    double w = 1;
    double b = 1;
    double sigma_w_gradient = 1;
    double sigma_b_gradient = 1;
    int count_y = 0;
    int count_x = 0;

    std::pair<double, double> learning_function(const std::vector<std::vector<double>> &x, const std::vector<int> &y)
    {
        for(int i = 0; i < 50000; i++)
        {

            sigma_w_gradient += (2 * std::pow(x[count_y][count_x],2) * w) + (2 * x[count_y][count_x] * b) - (2 * x[count_y][count_x] * y[count_y]);
            sigma_b_gradient += (2 * x[count_y][count_x] * w) - (2 * y[count_y]) + (2 * b);

            w = w - (0.0001*sigma_w_gradient);
            b = b - (0.0001*sigma_b_gradient);

            if(count_y == (x.size() - 1) && count_x == (x[0].size() - 1))
            {
                sigma_w_gradient /= 6;
                sigma_b_gradient /= 6;
            }

            if((i % ((x.size()*x[0].size()) - 1)) == 0)
            {
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
        
        for(int i = 0; i < 25000; i++)
        {
            sigma_w_gradient += -((y[count_y]- 1 / (1+pow(M_E, -(w * x_batch[count_y][count_x] + b)))) * x_batch[count_y][count_x]);
            sigma_b_gradient += -(y[count_y]- 1 / (1+pow(M_E, -(w * x_batch[count_y][count_x] + b))));
 
            w -= (0.000001*sigma_w_gradient);
            b -= (0.000001*sigma_b_gradient);

            if(count_y == (x_batch.size() - 1) && count_x == (x_batch[0].size() - 1))
            {
                sigma_w_gradient /= 200;
                sigma_b_gradient /= 200;
            }

            if((i % ((x_batch.size()*x_batch[0].size()) - 1)) == 0)
            {
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
    //행x열
    int row = 2;
    int column = 100; 
    double count = 1.0;

    std::vector<std::vector<double>> x_batch(row,std::vector<double>(column,0));

    for(int i = 0; i < row; i++)
    {
        for(int z = 0; z < column; z++)
        {
            x_batch[i][z] = count;

            count += 0.5;
        }
    }

    std::ofstream outputFile("data.txt");

    if (outputFile.is_open()) {
        for (int i = 0; i < row; i++) {
            for (int z = 0; z < column; z++) {
                outputFile << "(" << x_batch[i][z] << "," << i << ")" << "\n";
            }
            outputFile << "\n"; 
        }
        outputFile.close();
    } 

    DNN data_obj(x_batch,{0,1});

    std::cout << "Linear Regression  : " << std::endl;
    std::cout << "Equation: y = " << data_obj.output_linear_regression.first << "x + " << data_obj.output_linear_regression.second <<std::endl;
    std::cout << "Final cost : " << data_obj.cost << std::endl << std::endl;

    std::cout << "Logistic Regression  : " << std::endl;
    std::cout << "Equation: y = " << data_obj.output_logistic_regression.first << "x + " << data_obj.output_logistic_regression.second <<std::endl;
    std::cout << "Final cost : " << data_obj.logistic_cost << std::endl << std::endl;

    return 0;
}
