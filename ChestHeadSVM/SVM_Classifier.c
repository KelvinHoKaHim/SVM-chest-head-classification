#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define N_FEATURES 6
#define N_CLASSES 2
#define N_VECTORS 15
#define N_ROWS 2
#define N_COEFFICIENTS 1
#define N_INTERCEPTS 1
#define KERNEL_TYPE 'r'
#define KERNEL_GAMMA 0.02782957186431383
#define KERNEL_COEF 0.0
#define KERNEL_DEGREE 3

double vectors[15][6] = {{0.000235939742634, 0.7852085404878651, 2.736051012451508, 5.805195753237731, 1.2587609470027783, 2.2058426158873545}, {0.0003252998243282, 1.4621435202504662, 3.9588638040057726, 3.635688714425986, 1.5436020336380587, 1.8245632207034217}, {0.0001467493223901, 1.0892259482573183, 2.307253333064528, 6.099208465671034, 1.3126448131223405, 2.1497788437148277}, {0.0003850958894841, 1.3309280895194748, 4.722478615219197, 3.89136361194671, 1.4909366966787916, 1.8263529380802308}, {0.0002400089290362, 1.3173890216998905, 3.919682877816833, 4.232605602901644, 1.4228335929976337, 1.9598908303199527}, {0.0003009344190555, 1.2926514654167838, 4.045231863611569, 4.035168717997911, 1.4695925211711782, 1.9217192712000009}, {0.0002418899501553, 1.331657591739992, 3.535019142370621, 4.3661592757373135, 1.4681946657391516, 1.96022168974616}, {0.0002290008727608, 1.7626690261477505, 6.8276222148102885, 4.148542249373519, 1.605331502979527, 1.6788087723632503}, {0.0001222833886843, 1.5405332689183853, 5.721239357756951, 5.891888296486089, 1.6385699171095334, 1.6489514615316592}, {5.52458848229143e-05, 2.610631531045903, 8.86685040829805, 17.435826532586816, 1.631038019857636, 1.596113441274155}, {0.0002563226752196, 2.1400606069428045, 5.760322779955184, 5.487188209194319, 1.689932623280637, 1.4484472435102291}, {0.0002758625966423, 2.2130706224762102, 6.081536822568615, 4.702267676461199, 1.6973851247192224, 1.4627879914658914}, {0.0002385850173427, 2.15410753606492, 5.883480663078431, 5.379280140668412, 1.6997007210929775, 1.4511716762616873}, {0.0002425046192322, 2.17829925015966, 5.920520991521525, 5.373809221267473, 1.6756813476206265, 1.466301054249317}, {0.0002119168332225, 1.921287864225335, 2.51892887175588, 7.651380878186406, 1.1797793635551703, 2.1082780915615333}};
double coefficients[1][15] = {{-0.8329172197647533, -1.0, -1.0, -1.0, -1.0, -1.0, -0.8800928212187339, 0.6786640565385634, 1.0, 0.44080071290253947, 1.0, 1.0, 1.0, 0.5935452715423841, 1.0}};
double intercepts[1] = {-0.49994686698655944};
int weights[2] = {7, 8};

int predict (double features[]) {
    int i, j, k, d, l;

    double kernels[N_VECTORS];
    double kernel;
    switch (KERNEL_TYPE) {
        case 'l':
            // <x,x'>
            for (i = 0; i < N_VECTORS; i++) {
                kernel = 0.;
                for (j = 0; j < N_FEATURES; j++) {
                    kernel += vectors[i][j] * features[j];
                }
                kernels[i] = kernel;
            }
            break;
        case 'p':
            // (y<x,x'>+r)^d
            for (i = 0; i < N_VECTORS; i++) {
                kernel = 0.;
                for (j = 0; j < N_FEATURES; j++) {
                    kernel += vectors[i][j] * features[j];
                }
                kernels[i] = pow((KERNEL_GAMMA * kernel) + KERNEL_COEF, KERNEL_DEGREE);
            }
            break;
        case 'r':
            // exp(-y|x-x'|^2)
            for (i = 0; i < N_VECTORS; i++) {
                kernel = 0.;
                for (j = 0; j < N_FEATURES; j++) {
                    kernel += pow(vectors[i][j] - features[j], 2);
                }
                kernels[i] = exp(-KERNEL_GAMMA * kernel);
            }
            break;
        case 's':
            // tanh(y<x,x'>+r)
            for (i = 0; i < N_VECTORS; i++) {
                kernel = 0.;
                for (j = 0; j < N_FEATURES; j++) {
                    kernel += vectors[i][j] * features[j];
                }
                kernels[i] = tanh((KERNEL_GAMMA * kernel) + KERNEL_COEF);
            }
            break;
    }

    int starts[N_ROWS];
    int start;
    for (i = 0; i < N_ROWS; i++) {
        if (i != 0) {
            start = 0;
            for (j = 0; j < i; j++) {
                start += weights[j];
            }
            starts[i] = start;
        } else {
            starts[0] = 0;
        }
    }

    int ends[N_ROWS];
    for (i = 0; i < N_ROWS; i++) {
        ends[i] = weights[i] + starts[i];
    }

    if (N_CLASSES == 2) {

        for (i = 0; i < N_VECTORS; i++) {
            kernels[i] = -kernels[i];
        }

        double decision = 0.;
        for (k = starts[1]; k < ends[1]; k++) {
            decision += kernels[k] * coefficients[0][k];
        }
        for (k = starts[0]; k < ends[0]; k++) {
            decision += kernels[k] * coefficients[0][k];
        }
        decision += intercepts[0];

        if (decision > 0) {
            return 0;
        }
        return 1;

    }

    double decisions[N_INTERCEPTS];
    double tmp;
    for (i = 0, d = 0, l = N_ROWS; i < l; i++) {
        for (j = i + 1; j < l; j++) {
            tmp = 0.;
            for (k = starts[j]; k < ends[j]; k++) {
                tmp += kernels[k] * coefficients[i][k];
            }
            for (k = starts[i]; k < ends[i]; k++) {
                tmp += kernels[k] * coefficients[j - 1][k];
            }
            decisions[d] = tmp + intercepts[d];
            d = d + 1;
        }
    }

    int votes[N_INTERCEPTS];
    for (i = 0, d = 0, l = N_ROWS; i < l; i++) {
        for (j = i + 1; j < l; j++) {
            votes[d] = decisions[d] > 0 ? i : j;
            d = d + 1;
        }
    }

    int amounts[N_CLASSES];
    for (i = 0, l = N_CLASSES; i < l; i++) {
        amounts[i] = 0;
    }
    for (i = 0; i < N_INTERCEPTS; i++) {
        amounts[votes[i]] += 1;
    }

    int classVal = -1;
    int classIdx = -1;
    for (i = 0; i < N_CLASSES; i++) {
        if (amounts[i] > classVal) {
            classVal = amounts[i];
            classIdx= i;
        }
    }
    return classIdx;

}

int main(int argc, const char * argv[]) {

    /* Features: */
    double features[argc-1];
    int i;
    for (i = 1; i < argc; i++) {
        features[i-1] = atof(argv[i]);
    }

    /* Prediction: */
    printf("%d", predict(features));
    return 0;

}
