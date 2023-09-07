#include <iostream>
#include <vector>
#include <cmath>
#include <utility> 
#include <fstream>

# define M_E 2.7182818284590452354 /* e */

class DNN
{
    public:
    std::pair<std::vector<double>, std::vector<double>> output_linear_regression;
    std::pair<std::vector<double>, std::vector<double>> output_logistic_regression;
    double cost[3]; //needs to be changed
    double logistic_cost[3]; //needs to be changed

    DNN(const std::vector<std::vector<double>> &x_batch_, const std::vector<std::vector<double>> &y_batch_)
    {
        x_batch = x_batch_;
        y_batch = y_batch_;
        w.assign(y_batch.size(),1);
        b.assign(y_batch.size(),1);
        output_linear_regression = DNN::learning_function(x_batch,y_batch);
        output_logistic_regression = DNN::logistic(output_linear_regression.first,output_linear_regression.second);

    }


    private:
    std::vector<std::vector<double>> x_batch;
    std::vector<std::vector<double>> y_batch;
    std::vector<double> w;
    std::vector<double> b;
    double sigma_w_gradient[3]; //needs to be changed
    double sigma_b_gradient[3]; //needs to be changed
    int count_y = 0;
    int count_x = 0;

    // int softmax_epoch = 499999984;
    int softmax_epoch = 10000000;
    int percentage;

    std::pair<std::vector<double>, std::vector<double>> learning_function(const std::vector<std::vector<double>> &x, const std::vector<std::vector<double>> &y)
    {
        for(int i = 0; i < 3; i++)
        {
            for(int j = 0; j < 999991; j++) //needs to be changed 9965
            {
                sigma_w_gradient[i] += (2 * std::pow(x[count_y][count_x], 2) * w[i]) + (2 * x[count_y][count_x] * b[i]) - (2 * x[count_y][count_x] * y[i][count_y]);
                sigma_b_gradient[i] += (2 * x[count_y][count_x] * w[i]) - (2 * y[i][count_y]) + (2 * b[i]);

                if(count_y == (x.size() - 1) && count_x == (x[0].size() - 1))
                {
                    sigma_w_gradient[i] /= x.size() * x[0].size();
                    sigma_b_gradient[i] /= x.size() * x[0].size();
                    w[i] -= (0.000001*sigma_w_gradient[i]);
                    b[i] -= (0.000001*sigma_b_gradient[i]);
                    sigma_w_gradient[i] *= 0;
                    sigma_b_gradient[i] *= 0;
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
                cost[i] = (w[i] * x[count_y][count_x] + b[i]) - y[i][count_y];
            }

            count_y = 0;
            count_x = 0;
        }

        return std::make_pair(w,b);
    }

    std::pair<std::vector<double>, std::vector<double>> logistic(std::vector<double> w, std::vector<double> b)
    {
        int count_y = 0;
        int count_x = 0;

        for(int i = 0; i < 3; i++)
        {
            for(int j = 0; j < softmax_epoch; j++) //needs to be changed "j"
            {
                sigma_w_gradient[i] += -((y_batch[i][count_y] - 1 / (1+pow(M_E, -(w[i] * x_batch[count_y][count_x] + b[i])))) * x_batch[count_y][count_x]);
                sigma_b_gradient[i] += -(y_batch[i][count_y] - 1 / (1+pow(M_E, -(w[i] * x_batch[count_y][count_x] + b[i]))));
    
                if(j % 59 == 0)
                {
                    sigma_w_gradient[i] /= x_batch.size() * x_batch[0].size();
                    sigma_b_gradient[i] /= x_batch.size() * x_batch[0].size();

                    w[i] -= (0.001*sigma_w_gradient[i]);
                    b[i] -= (0.001*sigma_b_gradient[i]);

                    sigma_w_gradient[i] = 0;
                    sigma_b_gradient[i] = 0;
                }

                if((j % 10000000) == 0)
                {   
                    percentage = (static_cast<double>(j) / softmax_epoch) * 100;
                    // std::cout << percentage << " w : " << w[i] << "           " << "b : " << b[i] << "           "  << "cost :" <<logistic_cost[i] << std::endl << std::endl;
                    std::cout << "Node number[" << i << "] " << percentage << "%        w : " << w[i] << " b : " << b[i] << " cost : " << logistic_cost[i] << std::endl;
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

                logistic_cost[i] = y_batch[i][count_y] - 1 / (1+pow(M_E, -(w[i] * x_batch[count_y][count_x] + b[i])));
            }

            count_y = 0;
            count_x = 0;
        }

        return std::make_pair(w,b);
    }
};


int main()
{
    int row = 3;
    int column = 20; 
    double count = 1.0;
    double output = 0;
    float num = 0;

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
        for (int i = 0; i < row; i++) 
        {
            outputFile << "Data" << i+1 << " "; 
            for (int z = 0; z < column; z++) 
            {
                outputFile << x_batch[i][z] << " ";
            }
            outputFile << "\n"; 
        }
    }
    
    DNN data_obj(x_batch,{{1,0,0},{0,1,0},{0,0,1}});

    std::cout << "Linear Regression  : " << std::endl << std::endl;

    for(int i = 0; i < 3; i++)
    {
        std::cout << "equation number" << i+1 << std::endl;
        std::cout << "Equation: y = " << data_obj.output_linear_regression.first[i] << "x + " << data_obj.output_linear_regression.second[i] <<std::endl;
        std::cout << "Final cost : " << data_obj.cost[i] << std::endl << std::endl;
    }

    std::cout << "Logistic Regression  : " << std::endl << std::endl;

    for(int i = 0; i < 3; i++)
    {
        std::cout << "equation number" << i+1 << std::endl;
        std::cout << "Equation: y = " << data_obj.output_logistic_regression.first[i] << "x + " << data_obj.output_logistic_regression.second[i] <<std::endl;
        std::cout << "Final cost : " << data_obj.logistic_cost[i] << std::endl << std::endl;
    }

    // std::cout << "Softmax regression : " << std::endl << std::endl;
    // std::cout << "enter data" << std::endl;

    // std::cin >> num;

    // for(int i = 0; i < 3; i++)
    // {
    //     output = 1 / (1 + pow(M_E, -((data_obj.output_logistic_regression.first[i] * num) + data_obj.output_logistic_regression.second[i])));

    //     std::cout << i+1 << " : " << output*100 << "%" << std::endl;
    // }
    if (outputFile.is_open()) 
    {
        for(int i = 0; i < 3; i++)
        {
            outputFile << data_obj.output_logistic_regression.first[i] << " " <<data_obj.output_logistic_regression.second[i] << "\n";
        }
        outputFile.close();
    }
    return 0;
}
