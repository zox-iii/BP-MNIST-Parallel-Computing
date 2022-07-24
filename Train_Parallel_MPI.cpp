// Title:基于BP神经网络的MNIST手写体数字并行化实现
// Author:ZHANG GUOYANG、HE JIAER、ZHANG HONGMEI
// Data:2022.04.15

#include <iostream>     // 输入输出流
#include <fstream>      // 文件读写
#include <ctime>        // 计算程序运行时长 clock()函数
#include <cmath>        // 基本数学函数     exp()函数
#include <cstdlib>      // 标准函数库       rand()函数
#include <mpi.h>        // MPI并行计算

using namespace std;    // 标准命名空间

// +-------------+
// | 定义一些参数 |
// +-------------+
#define TRAIN_IMAGES_FILE "data/train-images.idx3-ubyte"
#define TRAIN_LABELS_FILE "data/train-labels.idx1-ubyte"

#define TEST_IMAGES_FILE "data/t10k-images.idx3-ubyte"
#define TEST_LABELS_FILE "data/t10k-labels.idx1-ubyte"

#define WEIGHTS_FILE "Model-Neural-Network(MPI).dat"
#define TRAIN_REPORT_FILE "Training-Report(MPI).dat"
#define TEST_REPORT_FILE "Testing-Report(MPI).dat"

int SIZE = 10;                          // 每个工作节点的最大样本量

const int num_Training = 10000;         // 训练样本总数
const int num_Testing = 10000;          // 测试样本总数
const int rows = 28;                    // 图像大小
const int cols = 28;                    // 图像大小
const int n1 = rows * cols;             // 输入层神经元数目
const int n2 = 30;                      // 隐藏层神经元数目
const int n3 = 10;                      // 输出分类类别共10个
const int epochs = 512;                 // 最大迭代次数
const double learning_rate = 1e-3;      // 学习率
const double momentum = 0.9;            // 动量
const double epsilon = 1e-3;            // 误差阈值

// in:输入、out:输出、w:权重矩阵、theta:输出层的误差、delta:整体误差对w的偏导
double w1[n1][n2], delta1[n1][n2], out1[n1];                 //输入层到隐藏层
double w2[n2][n3], delta2[n2][n3], in2[n2], out2[n2], theta2[n2];  //隐藏层到输出层
double in3[n3], out3[n3], theta3[n3], target[n3];               //输出层
double global_w1[n1][n2], global_w2[n2][n3];        // 全局权重
double res, loss;    // 总的损失
int predict, num_Correct;   // 预测值(0~9)、正确样本个数
// 所有样本数据
double image_data[num_Training][n1];                //图片数据
double label_data[num_Training][n3];                //标签数据
double test_image_data[num_Testing][n1];            //测试图片数据
double test_label_data[num_Testing][n3];            //测试标签数据
int target_number[num_Testing];                     //测试标签(0~9)
// 定义文件流
ifstream image;                                     //图片文件流
ifstream label;                                     //标签文件流
ofstream report;                                    //报告文件流
ofstream weights;                                   //模型权重文件流
ifstream load;
// +-----------------+
// | 输出模型相关信息 |
// +-----------------+
void show_details(){
    cout << "***********************************************************" << endl;
	cout << "*** Training Neural Network for MNIST database (Serial) ***" << endl;
    cout << "***********************************************************" << endl << endl;
    cout << "--(1)MNIST parameters" << endl;
    cout << "Training image data: " << TRAIN_IMAGES_FILE << endl;
	cout << "Training label data: " << TRAIN_LABELS_FILE << endl;
    cout << "Training sample: " << num_Training << endl;
    cout << "Image size: " << rows << "*" << cols << endl << endl;
    cout << "--(2)Network parameters" << endl;
    cout << "Input neurons: " << n1 << endl;
	cout << "Hidden neurons: " << n2 << endl;
	cout << "Output neurons: " << n3 << endl  << endl;
    cout << "--(3)Model training parameters" << endl;
    cout << "Max iterations: " << epochs << endl;
	cout << "Learning Rate: " << learning_rate << endl;
	cout << "Momentum: " << momentum << endl;
	cout << "Epsilon: " << epsilon << endl;
    cout << "Weights data: " << WEIGHTS_FILE << endl;
	cout << "Report data: " << TRAIN_REPORT_FILE << endl;
}

// +-------------------------------------+
// | 初始化网络模型中的输入、输出和权重矩阵 |
// +-------------------------------------+
void init_matrix(){
    res, loss = 0.0;
    predict, num_Correct = 0;
    // 初始化权重为0(?会导致对称权重现象)
    // 随机化权重(这里的随机化策略是均匀分布随机化)
    for(int i = 0; i < n1; i++){
        for(int j = 0; j < n2; j++){
            int temp = rand() % 2;//得到0~1之间的随机数
            w1[i][j] = (double)(rand() % 6) / 10.0;
            if(temp == 1){
                w1[i][j] = - w1[i][j];
            }
        }
    }
    for(int i = 0; i < n2; i++){
        for(int j = 0; j < n3; j++){
            int temp = rand() % 2;//得到0~1之间的随机数
            w2[i][j] = (double)(1 + rand() % 10) / (10.0 * n3);
            if(temp == 1){
                w2[i][j] = - w2[i][j];
            }
        }
    }
}

// +----------------------+
// | 读取所有训练样本的数据 |
// +----------------------+
void read_train_data(){
    image.open(TRAIN_IMAGES_FILE, ios::in | ios::binary); // 打开文件流（读入、二进制）
    label.open(TRAIN_LABELS_FILE, ios::in | ios::binary); // 打开文件流（读入、二进制）
    // 读数据集的head(可改进、暂时不用)
    char number;    // char是1个字节
    for (int i = 1; i <= 16; ++i) {
        image.read(&number, sizeof(char));  // 16个字节的head
	}
    for (int i = 1; i <= 8; ++i) {
        label.read(&number, sizeof(char));  // 8个字节的head
	}
    // 读数据集的data
    for(int sample = 0; sample < num_Training; sample++){
        // 读取图片
        for(int i = 0; i < rows; i++){
            for(int j = 0; j < cols; j++){
                int pos = i*rows + j;
                image.read(&number, sizeof(char));//
                if(number == 0){
				    image_data[sample][pos] = 0.0; //得到第一层的输出out1(所有样本的)
			    }else{
				    image_data[sample][pos] = 1.0;
			    }
            }
	    }
        // 读取标签
        label.read(&number, sizeof(char));
        for(int i = 0; i < n3; i++){
            label_data[sample][i] = 0.0;
        }
        label_data[sample][number] = 1.0;
    }
    image.close();// 关闭文件流
    label.close();// 关闭文件流
}

// +----------------------+
// | 读取所有测试样本的数据 |
// +----------------------+
void read_test_data(){
    image.open(TEST_IMAGES_FILE, ios::in | ios::binary); // 打开文件流（读入、二进制）
    label.open(TEST_LABELS_FILE, ios::in | ios::binary); // 打开文件流（读入、二进制）
    // 读数据集的head(可改进、暂时不用)
    char number;    // char是1个字节
    for (int i = 1; i <= 16; ++i) {
        image.read(&number, sizeof(char));  // 16个字节的head
	}
    for (int i = 1; i <= 8; ++i) {
        label.read(&number, sizeof(char));  // 8个字节的head
	}
    // 读数据集的data
    for(int sample = 0; sample < num_Testing; sample++){
        // 读取图片
        for(int i = 0; i < rows; i++){
            for(int j = 0; j < cols; j++){
                int pos = i*rows + j;
                image.read(&number, sizeof(char));//
                if(number == 0){
				    test_image_data[sample][pos] = 0.0; //得到第一层的输出out1(所有样本的)
			    }else{
				    test_image_data[sample][pos] = 1.0;
			    }
            }
	    }
        // 读取标签
        label.read(&number, sizeof(char));
        for(int i = 0; i < n3; i++){
            test_label_data[sample][i] = 0.0;
        }
        test_label_data[sample][number] = 1.0;
        target_number[sample] = int(number);
    }
    image.close();// 关闭文件流
    label.close();// 关闭文件流
}

// +-----------------+
// | 保存模型权重矩阵 |
// +-----------------+
void save_weights() {
    weights.open(WEIGHTS_FILE, ios::out);
	// 输入层到隐藏层
    for(int i = 0; i < n1; i++){
        for(int j = 0; j < n2; j++){
			weights << w1[i][j] << " ";
		}
		weights << endl;
    }
	// 隐藏层到输出层
    for(int i =0; i < n2; i++){
        for(int j =0; j < n3; j++){
			weights << w2[i][j] << " ";
		}
        weights << endl;
    }
	weights.close();//关闭文件流
}

// +-----------------+
// | 读取模型权重矩阵 |
// +-----------------+
void load_weights() {
    load.open(WEIGHTS_FILE, ios::in);
	// 输入层到隐藏层
    for(int i = 0; i < n1; i++){
        for(int j = 0; j < n2; j++){
			load >> w1[i][j];
		}
    }
	// 隐藏层到输出层
    for(int i =0; i < n2; i++){
        for(int j =0; j < n3; j++){
			load >> w2[i][j];
		}
    }
	load.close();//关闭文件流
}

// +------------------+
// | Sigmoid  激活函数 |
// +------------------+
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// +----------------------+
// | Square error 损失函数 |
// +----------------------+
double square_error(){
    res = 0.0;
    for(int i = 0; i < n3; i++){
        res += (out3[i] - target[i]) * (out3[i] - target[i]);
	}
    res *= 0.5;
    return res;
}

// +-----------------+
// | Accuracy 准确率 |
// +-----------------+
bool accuracy(){
    predict = 0;
    for(int i = 0; i < n3; i++){
        if(out3[i] > out3[predict]){
            predict = i;        // 比较得到最大的那个索引就是分类
        }
    }
    if(target[predict] == 1.0){
        num_Correct++;          // 预测正确个数加一
        return true;
    }
    return false;               // 返回该样本预测结果
}

// +---------+
// | 正向传播 |
// +---------+
void forward_process(){
    // 初始化in2、in3为0(写在这里，因为每一次迭代都需要更新为0)
    for(int i = 0; i < n2; i++){ in2[i] = 0.0; }
    for(int i = 0; i < n3; i++){ in3[i] = 0.0; }
    // 计算in2、out2
    for(int i = 0; i < n1; i++){
        for(int j =0; j < n2; j++){
            in2[j] += out1[i] * w1[i][j];
        }
    }
    for(int i = 0; i < n2; i++){ out2[i] = sigmoid(in2[i]); }
    // 计算in3、out3
    for(int i = 0; i < n2; i++){
        for(int j =0; j < n3; j++){
            in3[j] += out2[i] * w2[i][j];
        }
    }
    for(int i = 0; i < n3; i++){ out3[i] = sigmoid(in3[i]); }
}

// +---------+
// | 反向传播 |
// +---------+
void backward_process(){
    // 计算输出层误差
    for(int i = 0; i < n3; i++){
        theta3[i] = out3[i] * (1 - out3[i]) * (target[i] - out3[i]);
    }
    // 更新隐藏层到输出层的权重矩阵w2
    for(int i = 0; i < n2; i++){
        for(int j =0; j < n3; j++){
            delta2[i][j] = (learning_rate * theta3[j] * out2[i]) + (momentum * delta2[i][j]);
            w2[i][j] += delta2[i][j];
        }
    }
    // 计算隐藏层误差
    for(int i = 0; i < n2; i++){
        theta2[i] = 0.0;
        for(int j = 0; j < n3; j++){
            theta2[i] += w2[i][j] * theta3[j];
		}
        theta2[i] = out2[i] * (1 - out2[i]) * theta2[i];
    }
    // 更新输入层到隐藏层的权重矩阵w1
    for(int i = 0; i < n1; i++){
        for(int j =0; j < n2; j++){
            delta1[i][j] = (learning_rate * theta2[j] * out1[i]) + (momentum * delta1[i][j]);
            w1[i][j] += delta1[i][j];
        }
    }

}

// +-------------+
// | 样本训练过程 |
// +-------------+
int training_process(){
    // 初始化delta1、delta2为0
    for(int i = 0; i < n1; i++){
        for(int j =0; j < n2; j++){
            delta1[i][j] = 0.0;
        }
    }
    for(int i = 0; i < n2; i++){
        for(int j =0; j < n3; j++){
            delta2[i][j] = 0.0;
        }
    }
    // 开始迭代
    for(int epoch = 0; epoch < epochs; epoch++){
        // forward_process();//正向传播
        // 初始化in2、in3为0(写在这里，因为每一次迭代都需要更新为0)
        for(int i = 0; i < n2; i++){ in2[i] = 0.0; }
        for(int i = 0; i < n3; i++){ in3[i] = 0.0; }
        // 计算in2、out2
        for(int i = 0; i < n1; i++){
            for(int j =0; j < n2; j++){
                in2[j] += out1[i] * w1[i][j];
            }
        }
        for(int i = 0; i < n2; i++){ out2[i] = sigmoid(in2[i]); }
        // 计算in3、out3
        for(int i = 0; i < n2; i++){
            for(int j =0; j < n3; j++){
                in3[j] += out2[i] * w2[i][j];
            }
        }
        for(int i = 0; i < n3; i++){ out3[i] = sigmoid(in3[i]); }
        // backward_process();//反向传播
        // 计算输出层误差
        for(int i = 0; i < n3; i++){
            theta3[i] = out3[i] * (1 - out3[i]) * (target[i] - out3[i]);
        }
        // 更新隐藏层到输出层的权重矩阵w2
        for(int i = 0; i < n2; i++){
            for(int j =0; j < n3; j++){
                delta2[i][j] = (learning_rate * theta3[j] * out2[i]) + (momentum * delta2[i][j]);
                w2[i][j] += delta2[i][j];
            }
        }
        // 计算隐藏层误差
        for(int i = 0; i < n2; i++){
            theta2[i] = 0.0;
            for(int j = 0; j < n3; j++){
                theta2[i] += w2[i][j] * theta3[j];
            }
            theta2[i] = out2[i] * (1 - out2[i]) * theta2[i];
        }
        // 更新输入层到隐藏层的权重矩阵w1
        for(int i = 0; i < n1; i++){
            for(int j =0; j < n2; j++){
                delta1[i][j] = (learning_rate * theta2[j] * out1[i]) + (momentum * delta1[i][j]);
                w1[i][j] += delta1[i][j];
            }
        }
        res = square_error();//计算单个样本的损失
        if( res < epsilon){
            return epoch;// 返回迭代终止次数
        }
    }
    return epochs;
}

// +---------+
// | 测试过程 |
// +---------+
void test(){
	cout << endl << "*** Testing parameters ***" << endl;
    cout << "Testing image data: " << TRAIN_IMAGES_FILE << endl;
	cout << "Testing label data: " << TRAIN_LABELS_FILE << endl;
    cout << "Testing sample: " << num_Testing << endl;
	cout << "Report data: " << TEST_REPORT_FILE << endl;
    report.open(TEST_REPORT_FILE, ios::out);     // 打开文件流
    read_test_data();      // 获取所有测试样本数据
    clock_t begin = clock();                // 开始时间
    //load_weights();     // 获取模型权重矩阵w1、w2
    res, loss = 0.0;        // 参数清零
    predict, num_Correct = 0;
    for(int sample = 0;sample < num_Testing; sample++){
        for(int i = 0; i < n1; i++){ out1[i] = test_image_data[sample][i]; }     // 获取单个样本图片
        for(int i = 0; i < n3; i++){ target[i] = test_label_data[sample][i]; }   // 获取单个样本标签
        forward_process();      // 得到结果out3
        res = square_error();//计算单个样本的损失
        loss += res;      // 计算总的测试损失
        report << "Test Sample " << sample <<": Is the pre-correct = " << accuracy() << ", Label = " << target_number[sample] <<", Pre_Label = "<< predict <<  ", Error = " << res << endl;
        cout << "\r" << "Testing...";    
    }
    clock_t end = clock();                  //结束时间
    double running_time = double(end - begin) / CLOCKS_PER_SEC; //计算运行时间(秒)
    report << endl << "Total running time: " << running_time / 60 << " min" << endl;        // 总训练时间
    report << "Running time of a sample: " << running_time / num_Testing << " s" << endl;  // 单个样本训练时间
    report << "Test loss: " << loss / num_Testing << endl;     // 训练损失
    report << "Test accuracy: " << (double(num_Correct) / num_Testing)*100.0 << " %" << endl;     // 训练准确率
    report.close();                         // 关闭文件流
    cout << endl << "Testing is over!" << endl;          //测试完成标志
    cout << "Total running time: " << running_time / 60 << " min" << endl;
}

// +--------+
// | 主程序 |
// +--------+
int main(int argc, char *argv[]){  
    int rank, group_size, tag = 0;              // MPI参数
    double begin, end, running_time = 0.0;      // 时间参数
    MPI_Init(&argc, &argv);                     // MPI初始化
    MPI_Status status;                                  // 消息状态
    // 获取MPI信息
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);               // 获取进程号
    MPI_Comm_size(MPI_COMM_WORLD, &group_size);         // 获取进程组大小（进程数）
    cout << " This is rank : " << rank << " of " << group_size <<endl;      // 输出进程号
    group_size--;                                       // 获取用于训练的进程数(除去主进程)
    if(rank == 0){/*主进程开始计时*/
        report.open(TRAIN_REPORT_FILE, ios::out);   // 打开文件流
        show_details();                             // 输出详细信息
        begin = MPI_Wtime();                 // 开始时间
    }
    read_train_data();// 获取所有训练样本数据
    init_matrix();// 初始化模型矩阵参数  
    for(int i = 0;i < num_Training; i = i + group_size*SIZE){
        if(rank != 0){/*子进程训练样本*/
            int sample_num = 0;         // 每个工作节点的实际样本量
            if(i + group_size*SIZE > num_Training){/*当剩余样本量不足分配时*/
                sample_num = (num_Training - i) / group_size;
                if(rank <= (num_Training - i) % group_size){
                    sample_num++;
                }
            }
            else{
                sample_num = SIZE;  /*剩余样本量足够分配下一轮时，使用最大样本量SIZE*/
            }
            for(int loop = 0; loop < sample_num; loop++){
                int sample = i + ((rank - 1)*sample_num) + loop;        // 获取训练的样本号
                for(int i = 0; i < n1; i++){ out1[i] = image_data[sample][i]; }     // 获取单个样本图片
                for(int i = 0; i < n3; i++){ target[i] = label_data[sample][i]; }   // 获取单个样本标签
                int nIterations; //= training_process();//单个样本训练=正向传播+反向训练
                // 初始化delta1、delta2为0
                for(int i = 0; i < n1; i++){
                    for(int j =0; j < n2; j++){
                        delta1[i][j] = 0.0;
                    }
                }
                for(int i = 0; i < n2; i++){
                    for(int j =0; j < n3; j++){
                        delta2[i][j] = 0.0;
                    }
                }
                // 开始迭代
                for(int epoch = 0; epoch < epochs; epoch++){
                    // forward_process();//正向传播
                    // 初始化in2、in3为0(写在这里，因为每一次迭代都需要更新为0)
                    for(int i = 0; i < n2; i++){ in2[i] = 0.0; }
                    for(int i = 0; i < n3; i++){ in3[i] = 0.0; }
                    // 计算in2、out2
                    for(int i = 0; i < n1; i++){
                        for(int j =0; j < n2; j++){
                            in2[j] += out1[i] * w1[i][j];
                        }
                    }
                    for(int i = 0; i < n2; i++){ out2[i] = sigmoid(in2[i]); }
                    // 计算in3、out3
                    for(int i = 0; i < n2; i++){
                        for(int j =0; j < n3; j++){
                            in3[j] += out2[i] * w2[i][j];
                        }
                    }
                    for(int i = 0; i < n3; i++){ out3[i] = sigmoid(in3[i]); }
                    // backward_process();//反向传播
                    // 计算输出层误差
                    for(int i = 0; i < n3; i++){
                        theta3[i] = out3[i] * (1 - out3[i]) * (target[i] - out3[i]);
                    }
                    // 更新隐藏层到输出层的权重矩阵w2
                    for(int i = 0; i < n2; i++){
                        for(int j =0; j < n3; j++){
                            delta2[i][j] = (learning_rate * theta3[j] * out2[i]) + (momentum * delta2[i][j]);
                            w2[i][j] += delta2[i][j];
                        }
                    }
                    // 计算隐藏层误差
                    for(int i = 0; i < n2; i++){
                        theta2[i] = 0.0;
                        for(int j = 0; j < n3; j++){
                            theta2[i] += w2[i][j] * theta3[j];
                        }
                        theta2[i] = out2[i] * (1 - out2[i]) * theta2[i];
                    }
                    // 更新输入层到隐藏层的权重矩阵w1
                    for(int i = 0; i < n1; i++){
                        for(int j =0; j < n2; j++){
                            delta1[i][j] = (learning_rate * theta2[j] * out1[i]) + (momentum * delta1[i][j]);
                            w1[i][j] += delta1[i][j];
                        }
                    }
                    // res = square_error();//计算单个样本的损失
                    res = 0.0;
                    for(int i = 0; i < n3; i++){
                        res += (out3[i] - target[i]) * (out3[i] - target[i]);
                    }
                    res *= 0.5;
                    if( res < epsilon){
                        nIterations = epoch;// 返回迭代终止次数
                        break;
                    }
                    if(epoch == (epochs-1)){
                        nIterations = epochs;// 返回迭代最大次数
                        break;
                    }
                }
                loss += res;
                // 计算损失、预测结果、输出到report中
                cout << "Sample " << sample << ": No. iterations = " << nIterations << ", Error = " << res << ", Is the pre-correct = " << accuracy() << endl;  
                //report << "Sample " << sample << ": No. iterations = " << nIterations << ", Error = " << res << ", Is the pre-correct = " << accuracy() << endl;   
            }
        }
        if(rank != 0){/*子进程向主进程发送权重*/
            cout << "process " << rank << " send data w1 w2 to process 0" << endl;
            MPI_Send(w1, n1*n2, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD);
            MPI_Send(w2, n2*n3, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD);
        }
        if(rank == 0){/*主进程向子进程发送全局权重*/
            // 初始化global w1、w2
            for(int i = 0; i < n1; i++){
                for(int j = 0; j < n2; j++){
                    global_w1[i][j] = 0.0;        // 同步
                }
            }
            for(int i =0; i < n2; i++){
                for(int j = 0; j < n3; j++){
                    global_w2[i][j] = 0.0;        // 同步
                }
            }
            // 接收权重
            for(int j = 1; j <= group_size; j++){/*从1-11进程接收data*/
                // 接受消息
                cout << "process 0 recvive data w1 w2 from process " << j << endl;
                MPI_Recv(w1, n1*n2, MPI_DOUBLE, j, tag, MPI_COMM_WORLD, &status);
                MPI_Recv(w2, n2*n3, MPI_DOUBLE, j, tag, MPI_COMM_WORLD, &status);
                // 多个进程的权重求和
                for(int i = 0; i < n1; i++){
                    for(int j = 0; j < n2; j++){
                            global_w1[i][j] += w1[i][j];        // 同步
                    }
                }
                for(int i =0; i < n2; i++){
                    for(int j = 0; j < n3; j++){
                            global_w2[i][j] += w2[i][j];        // 同步
                    }
                }
            }
            // 平均权重
            for(int i = 0; i < n1; i++){
                for(int j = 0; j < n2; j++){
                    w1[i][j] = global_w1[i][j] / group_size;        
                }
            }
            for(int i =0; i < n2; i++){
                for(int j = 0; j < n3; j++){
                    w2[i][j] = global_w2[i][j] / group_size;        
                }
            }
            // 将平均后的权重再分发给各个子进程
            for(int j = 1; j <= group_size; j++){
                cout << "process 0 send data w1 w2 to process " << j << endl;
                MPI_Send(w1, n1*n2, MPI_DOUBLE, j, tag, MPI_COMM_WORLD);
                MPI_Send(w2, n2*n3, MPI_DOUBLE, j, tag, MPI_COMM_WORLD);
            }
        }
        if(rank != 0){/*子进程从主进程接收全局权重*/
            cout << "process " << rank << " recvive data w1 w2 from process 0" << endl;
            MPI_Recv(w1, n1*n2, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD, &status);
            MPI_Recv(w2, n2*n3, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD, &status);
        }
    }
    int gnum_Correct;
    double gloss;
    MPI_Reduce(&loss, &gloss, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);// 归约损失
    MPI_Reduce(&num_Correct, &gnum_Correct, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);// 归约正确个数，计算正确率
    if(rank == 0){/*主进程输出相关信息*/
        end = MPI_Wtime();                  //结束时间
        running_time += (end - begin); //计算运行时间(秒)
        // cout << running_time << " s " << endl;
        report << endl << "Total running time: " << running_time / 60 << " min" << endl;        // 总训练时间
        report << "Running time of a sample: " << running_time / num_Training << " s" << endl;  // 单个样本训练时间
        report << "Train loss: " << gloss / num_Training << endl;     // 训练损失
        report << "Train accuracy: " << (double(gnum_Correct) / num_Training)*100.0 << " %" << endl;     // 训练准确率
        report.close();                         // 关闭文件流
        save_weights();                         // 保存模型
        cout << endl << "Model training is over!" << endl;          //输出模型训练完成标志
        cout << "Total running time: " << running_time / 60 << " min" << endl;
        // 是否要测试
        int temp;
        cout << endl <<"if you want to test data,please input 1: ";
        cin >> temp;   // 输入一个数字
        if(temp == 1){
            test();
            cout << endl << "[Finish Train and Test]";
        }
        else{
            cout << endl << "[Finish Train]";
        }    // 归约损失
    }
    MPI_Finalize();                 // MPI结束
    return 0;
}