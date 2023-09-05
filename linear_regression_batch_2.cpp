#include <iostream>
#include <vector>
#include <cmath>
#include <utility> 

class DNN
{
    public:
    std::pair<double, double> output;
    double cost;

    DNN(const std::vector<std::vector<int>> &x_batch_, const std::vector<int> &y_)
    {
        x_batch = x_batch_;
        y = y_;
        output = DNN::learning_function(x_batch,y);
    }


    private:
    std::vector<std::vector<int>> x_batch;
    std::vector<int> y;
    double w = 1;
    double b = 1;
    double sigma_w_gradient = 1;
    double sigma_b_gradient = 1;
    int count = 0;
    int count_ = 0;

    std::pair<double, double> learning_function(const std::vector<std::vector<int>> &x, const std::vector<int> &y)
    {
        for(int i = 0; i < 10000; i++)
        {

            sigma_w_gradient += (2 * std::pow(x[count][count_],2) * w) + (2 * x[count][count_] * b) - (2 * x[count][count_] * y[count]);
            sigma_b_gradient += (2 * x[count][count_] * w) - (2 * y[count]) + (2 * b);

            w = w - (0.0001*sigma_w_gradient);
            b = b - (0.0001*sigma_b_gradient);

            if(count_ == 2 )
            {
                sigma_w_gradient /= 3;
                sigma_b_gradient /= 3;
            }

            if((((i+1) % 3)) == 0)
            {
                sigma_w_gradient = 0.0;
                sigma_b_gradient = 0.0;
            }

            // if((i % 500) == 0)
            // {
            //     std::cout << "w : " << w << "           " << "b : " << b << "           "  << "cost :" << (w * x[count][count_] + b) - y[count] <<std::endl;
            // }

            count_++;

            if(count_ == 3)
            {
                count_ = 0;
                count++;
                if(count == 5)
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
    int count = 1;
    std::vector<std::vector<int>> x_batch(5,std::vector<int>(3,0));

    for(int i = 0; i < 5; i++)
    {
        for(int z = 0; z < 3; z++)
        {
            x_batch[i][z] = count;

            count++;
            
        }

    }

    DNN data_obj(x_batch,{10,15,20,20,22});

    std::cout << "The equation of the learning is : y = " << data_obj.output.first << "x + " << data_obj.output.second <<std::endl;
    std::cout << "Final cost is : " << data_obj.cost << std::endl;

    return 0;
}
