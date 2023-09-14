#include <iostream>
#include <vector>
#include <cmath>
#include <utility> 
#include <fstream>

class DNN
{
    public:
    std::pair<double, double> output;
    double cost;

    DNN(const std::vector<int> &x_1, const std::vector<int> &x_2, const std::vector<int> &x_3, const std::vector<int> &y_,
         const int &epoch_, const float &lr_)
    {
        epoch = epoch_;
        lr = lr_;
        
        for(int i = 0; i < 3; i++)
        {
            x_all.push_back(x_1[i]);
            x_all.push_back(x_2[i]);
            x_all.push_back(x_3[i]);
            count_++;
        }

        y = y_;
        output = DNN::learning_function(x_all,y);
    }

    private:
    std::vector<int> x1;
    std::vector<int> x2;
    std::vector<int> x3;
    std::vector<int> x_all;
    std::vector<int> y;

    double w;
    double b;

    double sigma_w_gradient;
    double sigma_b_gradient;

    int count;
    int count_;

    int epoch;
    float lr;

    std::pair<double, double> learning_function(const std::vector<int> &x, const std::vector<int> &y)
    {
        for(int i = 0; i < epoch; i++)
        {
            sigma_w_gradient += (2 * std::pow(x[count],2) * w) + (2 * x[count] * b) - (2 * x[count] * y[count]);
            sigma_b_gradient += (2 * x[count] * w) - (2 * y[count]) + (2 * b);
            
            if(count == (count_-1))
            {
                sigma_w_gradient /= x.size();
                sigma_b_gradient /= x.size();

                w -= (lr * sigma_w_gradient);
                b -= (lr * sigma_b_gradient);

                sigma_w_gradient = 0;
                sigma_b_gradient = 0;
            }

            count++;

            if(count == count_)
            {
                count = 0;
            }

            // if((i % 500) == 0)
            // {
            //     std::cout << "w : " << w << "           " << "b : " << b << "           "  << "cost :" << (w * x[count] + b) - y[count] <<std::endl;
            // }

        }

        cost = (w * x[count] + b) - y[count];

        return std::make_pair(w,b);
    }

};

int main()
{
    int epoch = 20000;
    float lr = 0.1;

    int x = 1;
    int y = 10;

    std::ofstream outputFile("data/linear_regression_2_data.txt");

    for (int i = 0; i < 3; i++) 
    {
        for (int z = 0; z < 3; z++) 
        {
            outputFile << "(" << x << "," << y << ")" << "\n";
            x++;
        }
        y += 5;
    }

    DNN data_obj({1,2,3}, {4,5,6}, {7,8,9}, {10,15,20}, epoch, lr);

    std::cout << "Equation : y = " << data_obj.output.first << "x + " << data_obj.output.second <<std::endl;
    std::cout << "Final cost  : " << data_obj.cost << std::endl;

    outputFile << "\n" << "y = " << data_obj.output.first << "x + " << data_obj.output.second;

    return 0;
}
