#include "Main/Main.h"
#include <sys/types.h>
#include <vector>


int main() {

    struct Net : Module {

        Linear fc1{};

        Net() : fc1(2, 10) {
            register_module("fc1", fc1);
        }
        
        Tensor forward(Tensor& x) {
            return fc1(x);
        }
    };

    Net network;

    // Tensor A = Tensor(std::vector<double>({1,2,3,4,5,6,7,8,9,10}), std::vector<size_t>({1,5,2}), std::vector<size_t>({2,1}));
    // Tensor W = Tensor(std::vector<double>({1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20}), std::vector<size_t>({1,2,10}), std::vector<size_t>({20,10,1}));

    Tensor A = Tensor(std::vector<double>({1,2,3,4,5,6,7,8,9,10}), std::vector<size_t>({5,2}), std::vector<size_t>({2,1}));
    Tensor W = Tensor(std::vector<double>({1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20}), std::vector<size_t>({2,10}), std::vector<size_t>({10,1}));
    Tensor B = Tensor(std::vector<double>({1,2,3,4,5,6,7,8,9,10}), std::vector<size_t>({1, 10}), std::vector<size_t>({10, 1}));

    // Tensor C = (A & W);
    
    // Tensor D = C + B;
    
    // D.backward(true);

    // std::cout << D << std::endl << "HERE IS THE GE" << std::endl;
    // for (auto a : (*B.grad)) {
    //     std::cout << a;
    // };

   Tensor Y = network.forward(A);

    Y.backward(true);

    std::cout << Y << std::endl << std::endl;

    for (auto a : (*network.fc1.weights.grad)) {
        std::cout << a << ", ";
    } std::cout << std::endl << "isodfjpo" << std::endl;
    for (auto a : (*A.grad)) {
        std::cout << a << ", ";
    } std::cout << std::endl << "isodfjpo" << std::endl;


    // for (auto a : (*network.fc1.bias.grad)) {
    //     std::cout << a << ", ";
    // } std::cout << std::endl << std::endl;
}

// int main() {

//     struct Net : Module {

//         Linear fc1{}, fc2{};

//         Net() : fc1(2, 10), fc2(10, 1) {
//             register_module("fc1", fc1);
//         }

//         Tensor forward(Tensor& x) {
//             return fc1(x);
//         }
//     };

//     Net network;

//     Tensor A = Tensor(std::vector<double>({1,2,3,4,5,6,7,8,9,10}), std::vector<size_t>({5,2}), std::vector<size_t>({2,1}));

//     Tensor Y = network.forward(A);
// }


// int main() {
//     Tensor A = Tensor(  std::vector<double>({1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24}),
//                         std::vector<size_t>({2,3,4}),
//                         std::vector<size_t>({12,4,1}));
            
//     Tensor B = Tensor(  std::vector<double>({1,2,3,4,5,6,7,8}),
//                         std::vector<size_t>({1,4,2}),
//                         std::vector<size_t>({8,2,1}));

//     Tensor D = Tensor(  std::vector<double>({1,2,3,4,5,6}),
//                     std::vector<size_t>({1,2,3}),
//                     std::vector<size_t>({6,3,1}));

//     Tensor C = A & B; // (2,3,4) * (1,4,2) = (2,3,2)
//     Tensor E = C & D; // (2,3,2) * (1,2,3) = (2,3,3)

//     E.backward(true); // NOTE: this was just a quick fix to the graidnet keep on being set to 1 OOPSISE

//     std::cout << std::endl;
//     for (auto a : C.shape) {
//         std::cout << a << ", ";
//     }
//     std::cout << std::endl << std::endl;


//     for (auto a : (*A.grad)) {
//         std::cout << a << ",";
//     } std::cout << std::endl;
//     for (auto a : (*B.grad)) {
//         std::cout << a << ",";
//     } std::cout << std::endl;
//     for (auto a : (*C.grad)) {
//         std::cout << a << ",";
//     } std::cout << std::endl;
//     for (auto a : (*D.grad)) {
//         std::cout << a << ",";
//     } std::cout << std::endl;
//     for (auto a : (*E.grad)) {
//         std::cout << a << ",";
//     } std::cout << std::endl;
// }
