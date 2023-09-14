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

    DNN(const std::vector<std::vector<int>> &x_batch_, const std::vector<int> &y_, const int &epoch_, const float &lr_)
    {
        x_batch = x_batch_;
        y = y_;
        epoch = epoch_;
        lr = lr_;
        output = DNN::learning_function(x_batch,y);
    }


    private:
    std::vector<std::vector<int>> x_batch;
    std::vector<int> y;

    double w;
    double b;

    double sigma_w_gradient;
    double sigma_b_gradient;

    int count;
    int count_;

    int epoch;
    float lr;

    std::pair<double, double> learning_function(const std::vector<std::vector<int>> &x, const std::vector<int> &y)
    {
        for(int i = 0; i < epoch; i++)
        {
            sigma_w_gradient += (2 * std::pow(x[count][count_],2) * w) + (2 * x[count][count_] * b) - (2 * x[count][count_] * y[count]);
            sigma_b_gradient += (2 * x[count][count_] * w) - (2 * y[count]) + (2 * b);

            if(count_ == (x[0].size()-1))
            {
                sigma_w_gradient /= x[0].size();
                sigma_b_gradient /= x[0].size();

                w -= (lr * sigma_w_gradient);
                b -= (lr * sigma_b_gradient);

                sigma_w_gradient = 0;
                sigma_b_gradient = 0;
            }

            // if((i % 500) == 0)
            // {
            //     std::cout << "w : " << w << "           " << "b : " << b << "           "  << "cost :" << (w * x[count][count_] + b) - y[count] <<std::endl;
            // }

            count_++;

            if(count_ == x[0].size())
            {
                count_ = 0;
                count++;

                if(count == x.size())
                {
                    count = 0;
                }
            }

            cost = (w * x[count][count_] + b) - y[count];
        }

        return std::make_pair(w,b);
    }
};

int main()
{
    int epoch = 1000000;
    float lr = 0.001;

    int count = 1;

    std::vector<std::vector<int>> x_batch(5,std::vector<int>(3,0));
    std::vector<int> y_batch = {10,15,20,20,22};

    for(int i = 0; i < 5; i++)
    {
        for(int z = 0; z < 3; z++)
        {
            x_batch[i][z] = count;
            count++;
        }
    }

    std::ofstream outputFile("data/linear_regression_batch_2_data.txt");

    for (int i = 0; i < 5; i++) 
    {
        for (int z = 0; z < 3; z++) 
        {
            outputFile << "(" << x_batch[i][z] << "," << y_batch[i] << ")" << "\n";
        }
    }

    DNN data_obj(x_batch, y_batch, epoch, lr);

    std::cout << "Equation : y = " << data_obj.output.first << "x + " << data_obj.output.second <<std::endl;
    std::cout << "Final cost : " << data_obj.cost << std::endl;

    outputFile << "\n" << "y = " << data_obj.output.first << "x + " << data_obj.output.second;

    return 0;
}
