#include <iostream>
#include <vector>
#include <cmath>
#include <utility> 

class DNN
{
    public:
    std::pair<double, double> output;
    double cost = 0;

    DNN(const std::vector<int> &x_, const std::vector<int> &y_)
    {
        x = x_;
        y = y_;
        output = DNN::learning_function(x,y);
    }

    private:
    std::vector<int> x;
    std::vector<int> y;
    double w = 1;
    double b = 1;
    double sigma_w_gradient = 1;
    double sigma_b_gradient = 1;
    int count = 0;

    std::pair<double, double> learning_function(const std::vector<int> &x, const std::vector<int> &y)
    {
        for(int i = 0; i < 5000; i++)
        {
            sigma_w_gradient += (2 * std::pow(x[count],2) * w) + (2 * x[count] * b) - (2 * x[count] * y[count]);
            sigma_b_gradient += (2 * x[count] * w) - (2 * y[count]) + (2 * b);
            
            if(count == 2)
            {
                sigma_w_gradient /= 3.0;
                sigma_b_gradient /= 3.0;
                count -= 3;
            }

            count++;

            w = w - (0.01*sigma_w_gradient);
            b = b - (0.01*sigma_b_gradient);

            if(((i + 1) % 3) == 0)
            {
                sigma_w_gradient = 0.0;
                sigma_b_gradient = 0.0;
            }

            // if((i % 500) == 0)
            // {
            //     std::cout << "w : " << w << "           " << "b : " << b << "           "  << "cost :" << (w * x[count] + b) - y[count] <<std::endl;
            // }
            cost = (w * x[count] + b) - y[count];
        }

        return std::make_pair(w,b);
    }

};

int main()
{
    bool verify;

    DNN data_obj({1,2,3},{2,4,6});

    std::cout << "The equation of the learning is : y = " << data_obj.output.first << "x + " << data_obj.output.second <<std::endl;
    std::cout << "Final cost is : " << data_obj.cost << std::endl;

    return 0;
}
