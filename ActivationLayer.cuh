#pragma once

enum class ActivationType {
    ReLU,
    Sigmoid,
    Tanh,
};

class ActivationLayer {
    public:
        ActivationLayer(int in_dim, ActivationType type);
        ~ActivationLayer();

        void forward(float* input_d);

    private:
        int in_dim;
        ActivationType type;
};