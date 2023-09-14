#include <iostream>
#include <vector>
#include <cmath>
#include <utility> 
#include <fstream>

# define M_E 2.7182818284590452354 /* e */

int main()
{
    float num = 1.0;
    double output1 = 0;
    double output2 = 0;
    double output3 = 0;
    int predict = 0;
    int count = 0;
    while(true)
    {
        output1 = 1 / (1 + pow(M_E, -((-25.1908 * num)+270.778)));
        output2 = 1 / (1 + pow(M_E, -((-0.000156327 * num)-0.693365)));
        output3 = 1 / (1 + pow(M_E, -((1.46989 * num)-30.4452)));

        std::cout << num << std::endl << std::endl;
        std::cout << output1*100 << "%" << std::endl;
        std::cout << output2*100 << "%" << std::endl;
        std::cout << output3*100 << "%" << std::endl;

        if(output1>output2 && output1>output3)
        {
            predict = 1;
            std::cout << "Output : " << predict << std::endl;
        }

        else if(output2>output1 && output2>output3)
        {
            predict = 2;
            std::cout << "Output : " << predict << std::endl;
        }

        else if(output3>output1 && output3>output2)
        {
            predict = 3;
            std::cout << "Output : " << predict << std::endl;
        }
        else
        {
            std::cout<<"ERROR"<<std::endl;
        }

        if(0 < num && num < 11)
        {
            if(predict != 1)
            {
                count++;
            }
        }
        else if(10.5 < num && num < 21)
        {
            if(predict != 2)
            {
                count++;
            }
        }
        else if(20.5 < num && num < 31)
        {
            if(predict != 3)
            {
                count++;
            }
        }
        else
        {
            count = count;
        }

        std::cout << "=================" << std::endl << std::endl;
        num += 0.5;

        if(num == 31)
        {
            std::cout << "Prediction fail : " << count << "/60" << std::endl;
            std::cout << "Prediction accuracy : " << (static_cast<double>(31-count) / 31) * 100 << "%" << std::endl;
            break;
        }

    }
    return 0;
}

// equation number1
// Equation: y = -1.32463x + 14.1234
// Final cost : -4.11756e-08

// equation number2
// Equation: y = -0.000968025x + -0.693421
// Final cost : -0.328237

// equation number3
// Equation: y = 0.850095x + -17.5448
// Final cost : 0.0807313


// epoch 499999984
// lr 0.001
// equation number1
// Equation: y = -2.28494x + 24.4943
// Final cost : 2.18397e-08

// equation number2
// Equation: y = -0.000156327x + -0.693365
// Final cost : -0.333181

// equation number3
// Equation: y = 1.46989x + -30.4452
// Final cost : -4.93056e-12