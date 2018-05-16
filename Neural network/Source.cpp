#include<iostream>
#include<vector>
#include<algorithm>
#include<string>
#include<utility>
#include<exception>
#include<random>
#include<fstream>
#include<cmath>
using namespace std;

using LaerModel = vector<vector<double>>;

double InnerProduct(const vector<double>& left, const vector<double>& right)
{
	if (left.size() != right.size())
		throw exception("InnerProduct(): Error size");
	double InProduct = 0;
	for (size_t i = 0; i < left.size(); ++i)
	{
		InProduct += left[i] * right[i];
	}
	return InProduct;
}

class Model
{
private:
	string NameModel;
	vector<LaerModel> ModelNet;
public:
	void SetModelName(const string& name)
	{
		NameModel = name;
	}
	decltype(auto) operator[](size_t index)const
	{
		return ModelNet[index];
	}
	void InsertBackLaerModel(const LaerModel& lmodel)
	{
		ModelNet.emplace_back(lmodel);
	}
	size_t CountLaersInCurrentModel()const
	{
		return ModelNet.size();
	}
};

class IBaseFunction
{
public:
	virtual double operator()(double x) = 0;
	virtual double difCalculate(double x) = 0;
};

class LogisticFunc :public IBaseFunction
{
public:
	double difCalculate(double x) override
	{
		return (*this)(x)*(1 - (*this)(x));
	}
	double operator()(double x)
	{
		return 1 / (1 + exp(-x));
	}
};
class LinearFunc :public IBaseFunction
{
public:
	double difCalculate(double x)override
	{
		return 1;
	}
	double operator()(double x)
	{
		return x;
	}
};
class HTanFunc : public IBaseFunction
{
public:
	double operator()(double x)override
	{
		return tanh(x);
	}
	double difCalculate(double x)
	{
		return (1 - (*this)(x))*(1 + (*this)(x));
	}
};
class SoftPlusFunc :public IBaseFunction
{
public:
	double operator()(double x)override
	{
		return log(1 + exp(x));
	}
	double difCalculate(double x)override
	{
		return 1 / (1 + exp(-x));
	}
};



class Neiron
{
private:
	vector<double> weights;
	double localGrad;
	double difRez;
	double afRez;
public:
	Neiron() :weights(), localGrad(0), difRez(0), afRez(0) {}
	Neiron(const Neiron&) = default;
	Neiron(Neiron&&) = default;
	Neiron(const vector<double>& _weights) :weights(_weights) {}
	template<typename Func>
	double OutData(Func& _func, const vector<double>& _inputData) const
	{
		return _func(InnerProduct(_inputData, weights));
	}
	// ужно додумать
	void RandomInitWeights()
	{
		uniform_real_distribution<double> urd(-3, 3);
		random_device rd;
		for (auto& x : weights)
			x = urd(rd);
	}

	void InitWeights(const vector<double>& _weights)
	{
		weights = _weights;
	}

	decltype(auto) GetWeights()
	{
		return (weights);
	}

	double GetLocalGrad()const
	{
		return localGrad;
	}
	void SetLocalGrad(double lg)
	{
		localGrad = lg;
	}
	void SetDifRez(double df)
	{
		difRez = df;
	}
	double GetDifRez()const
	{
		return difRez;
	}
	void SetActiveRez(double afrez)
	{
		afRez = afrez;
	}
	double GetActiveRez()const
	{
		return afRez;
	}
};

class Layer
{
private:
	vector<double>InputData;
	vector<double>OutputData;
	vector<Neiron> laer;
	IBaseFunction* func;
public:
	Layer() :InputData(), OutputData(), laer(), func(nullptr) {}
	Layer(size_t count)
	{
		addNeirons(count);
	}
	// выставляем входные данные для слоя
	void SetInputData(const vector<double>& _input)
	{
		InputData = _input;
	}
	void SetInputData(vector<double>&& _input)
	{
		InputData = move(_input);
	}

	// возвращаем по ссылке выходные данные слоя
	decltype(auto) GetOutputDataRef()
	{
		return (OutputData);
	}

	// получаем доступ к отдельному нейрону слоя
	Neiron& operator[](size_t index)
	{
		return laer.at(index);
	}

	// прогоняем данные через весь слой и заносим их в OutputData
	void Forward()
	{
		if (!OutputData.empty())
		{
			OutputData.clear();
		}
		for (size_t i = 0; i < laer.size(); ++i)
		{
			double temp = laer[i].OutData(*func, InputData);
			laer[i].SetActiveRez(temp);
			OutputData.emplace_back(temp);
			laer[i].SetDifRez(laer[i].OutData([&](double a) {return func->difCalculate(a); }, InputData));
		}
	}

	/*void addFuncActivated(BaseActivFunction* bsFunc)
	{

	}*/

	// добавляем новые нейроны в количетсве count
	void addNeirons(size_t count)
	{
		for (size_t i = 0; i < count; ++i)
		{
			laer.emplace_back(Neiron());
		}
	}
	// добавляем функцию активации
	void addActivatedFunction(IBaseFunction* _func)
	{
		func = _func;
	}
	// добавляем производную функции активации
	/*void addDifActivatedFunction(IBaseFunction* _diffunc)
	{
		difFunc = _diffunc;
	}*/
	//количество нейронов
	size_t GetCountNeiron()const
	{
		return laer.size();
	}

	// инициализация нейронов
	void InitNeirons(size_t szw)
	{
		for (auto& x : laer)
		{
			x.GetWeights().resize(szw);
			x.RandomInitWeights();
		}
	}

	// возвращает функцию активации для слоя
};

class Net
{
private:
	vector<Layer> layers;
	double errorNet;
	vector<double> input;
	//vector<double> correctOutput;
	//Layer outLayer;
	vector<double> output;
	Model modelNet;
	size_t szInput;
	double loss;
public:
	Net() = default;
	void InitInputLayer(size_t _szInput)
	{
		szInput = _szInput;
	}
	void Forward(const vector<double>& dt)
	{
		input = dt;
		layers[0].SetInputData(dt);
		layers[0].Forward();
		for (size_t i = 1; i < layers.size(); ++i)
		{
			layers[i].SetInputData(move(layers[i - 1].GetOutputDataRef()));
			layers[i].Forward();
		}
		output = move(layers[layers.size() - 1].GetOutputDataRef());
	}
	void Backward()
	{

	}

	//обучение сети
	void Training(const vector<double>& trainData, const vector<double>& correctOutput, size_t epoch, double speedLn, double momentum = 0)
	{
		uniform_int_distribution<size_t> uid(0, (trainData.size() - 1) / szInput);
		random_device rd;
		//correctOutput = corOutput;
		for (size_t ep = 0; ep < epoch; ++ep)
		{
			for (size_t tr = 0; tr < trainData.size() / szInput; ++tr)
			{

				size_t index = tr;
				size_t pos = index*szInput;
				// прямой ход
				Forward(vector<double>(trainData.cbegin() + pos, trainData.cbegin() + szInput + pos));

				// вычисление ошибки сети
				errorNet = correctOutput[index] - output[0];

				// обратный ход

				//  для выходного слоя устанавливаем локальный градиент каждого нейрона

				for (size_t i = 0; i < layers[layers.size() - 1].GetCountNeiron(); ++i)
				{
					layers[layers.size() - 1][i].SetLocalGrad(errorNet*layers[layers.size() - 1][i].GetDifRez());
				}

				// вычисление локальных градиентов следующих скрытых слоев
				for (int64_t i = layers.size() - 2; i >= 0; --i)
				{
					for (size_t j = 0; j < layers[i].GetCountNeiron(); ++j)
					{
						double sum = 0;
						for (size_t k = 0; k < layers[i + 1].GetCountNeiron(); ++k)
						{
							double a = layers[i + 1][k].GetWeights()[j];
							double b = layers[i + 1][k].GetLocalGrad();
							sum += a*b;
						}
						layers[i][j].SetLocalGrad(sum*layers[i][j].GetDifRez());

					}
				}

				// корректировка весовых коэффицентов

				// для первого скрытого слоя
			/*	for (size_t w = 0; w < layers[0][j].GetWeights().size(); ++w)
					{*/
				for (size_t j = 0; j < input.size(); ++j)
				{
					for (size_t i = 0; i < layers[0].GetCountNeiron(); ++i)
					{
						double w = speedLn*input[j] * layers[0][i].GetLocalGrad();
						layers[0][i].GetWeights()[j] += w;
					}
				}

				// для остальных



				for (size_t l = 1; l < layers.size(); ++l)
				{
					for (size_t j = 0; j < layers[l].GetCountNeiron(); ++j)
					{
						for (size_t w = 0; w < layers[l][j].GetWeights().size(); ++w)
						{
							double dw = speedLn*layers[l - 1][w].GetActiveRez()*layers[l][j].GetLocalGrad();
							layers[l][j].GetWeights()[w] += dw;
						}
					}
				}

			}
		}


	}
	//void AddLaerAndInit(size_t count)
	//{
	//	for (size_t i = 0; i < count; ++i)
	//	{
	//		laers.emplace_back(Layer());
	//	}
	//}

	// задачется вектор размер которого равен количетсву слоев
	// каждый элемент вектора это количетсво нейронов в соотвтсвующем слою
	void SetingNet(const vector<size_t>& stngsHide, const vector<IBaseFunction*>& activFunctions)
	{
		// создали все скрытые слои в количетсве stngs.size() с нейронами внутри. количетсво в каждом слою stngs[i]
		for (size_t i = 0; i < stngsHide.size(); ++i)
		{
			layers.emplace_back(Layer(stngsHide[i]));
			layers[i].addActivatedFunction(activFunctions.at(i));
		}
		// связь с входным слоем и первым скрытым слоем
		if ((szInput == 0))
			throw exception("Don't init input layer");

		layers[0].InitNeirons(szInput);

		// выставляем связь между остальными скрытыми слоями
		for (size_t i = 1; i < layers.size(); ++i)
		{
			layers[i].InitNeirons(layers[i - 1].GetCountNeiron());
		}

		// создание выходного слоя


	}

	//Надо додумать
	/*void SaveModel(const string& nameModel = "model")
	{
		fstream file;
		modelNet.SetModelName(nameModel);
		file.open(nameModel + ".txt");
	}
	void InitNet(const string& nameModel)
	{
		fstream file;
		file.open(nameModel);
		if (!file.is_open())
		{
			throw exception("Not found model: may be uncorrect name file");
		}

	}*/
	decltype(auto) GetModelNetRef()
	{
		return modelNet;
	}

	// выход сети 
	decltype(auto) OutNet()const
	{
		return (output);
	}

	// возвращает вектор слоев
	decltype(auto) GetLayers()
	{
		return(layers);
	}
};




int main()
{

	Layer l1;
	l1.addActivatedFunction(new LogisticFunc());
	l1.addNeirons(4);
	l1[0].InitWeights(vector<double>{1, 2});
	l1[1].InitWeights(vector<double>{3, -2});
	l1[2].InitWeights(vector<double>{5, 2});
	l1[3].InitWeights(vector<double>{4, 8});

	Layer l2;
	l2.addActivatedFunction(new LogisticFunc());
	l2.addNeirons(3);
	l2[0].InitWeights(vector<double>{-1, 2, 2, 6});
	l2[1].InitWeights(vector<double>{1, 1, -4, 8, });
	l2[2].InitWeights(vector<double>{3, -2, 1, -9});

	Layer l3;
	l3.addActivatedFunction(new SoftPlusFunc());
	l3.addNeirons(1);
	l3[0].InitWeights(vector<double>{1, 2, 4});


	Net net;
	net.GetLayers().emplace_back(l1);
	net.GetLayers().emplace_back(l2);
	net.GetLayers().emplace_back(l3);
	net.InitInputLayer(2);

	vector<double> input{ 1,1,0,1,1,0,0,0 };
	vector<double>d{ 6,5,9,2 };

	net.Training(input, d, 150, 0.65);

	for (int i = 0; i < 4; i++)
	{

		vector<double>t(input.begin() + i * 2, input.begin() + 2 + i * 2);
		net.Forward(t);
		cout.precision(30);
		cout.setf(ios_base::fixed);
		cout << net.OutNet()[0] << endl;
	}

	/*vector<size_t> init = { 2,3,1 };
	vector<IBaseFunction*> funcs = { new LogisticFunc(),new LogisticFunc(),new LogisticFunc() };
	Net net;
	net.InitInputLayer(2);
	net.SetingNet(init, funcs);*/

	system("pause");
}