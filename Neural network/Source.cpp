#include<iostream>
#include<vector>
#include<algorithm>
#include<string>
#include<utility>
#include<exception>
#include<random>
#include<fstream>
#include<cmath>


//using LaerModel = vector<vector<double>>;
//
//void NonlinearNormal(vector<double>& vec)
//{
//	auto minElem=max_element(vec.begin(),vec.end());
//	auto maxElem=min_element(vec.begin(), vec.end());;
//	double XC = ((*maxElem) + (*minElem)) / 2;
//	for (auto& x : vec)
//	{
//		x = 1 / (1 + exp(-(x - XC)));
//	}
//}
//
//double InnerProduct(const vector<double>& left, const vector<double>& right)
//{
//	if (left.size() != right.size())
//		throw exception("InnerProduct(): Error size");
//	double InProduct = 0;
//	for (size_t i = 0; i < left.size(); ++i)
//	{
//		InProduct += left[i] * right[i];
//	}
//	return InProduct;
//}
//
//class Model
//{
//private:
//	string NameModel;
//	vector<LaerModel> ModelNet;
//public:
//	void SetModelName(const string& name)
//	{
//		NameModel = name;
//	}
//	decltype(auto) operator[](size_t index)const
//	{
//		return ModelNet[index];
//	}
//	void InsertBackLaerModel(const LaerModel& lmodel)
//	{
//		ModelNet.emplace_back(lmodel);
//	}
//	size_t CountLaersInCurrentModel()const
//	{
//		return ModelNet.size();
//	}
//};
//
//class IBaseFunction
//{
//public:
//	virtual double operator()(double x) = 0;
//	virtual double difCalculate(double x) = 0;
//};
//
//class LogisticFunc :public IBaseFunction
//{
//public:
//	double difCalculate(double x) override
//	{
//		return (*this)(x)*(1 - (*this)(x));
//	}
//	double operator()(double x)
//	{
//		return 1 / (1 + exp(-x));
//	}
//};
//class LinearFunc :public IBaseFunction
//{
//public:
//	double difCalculate(double x)override
//	{
//		return 1;
//	}
//	double operator()(double x)
//	{
//		return x;
//	}
//};
//class HTanFunc : public IBaseFunction
//{
//public:
//	double operator()(double x)override
//	{
//		return tanh(x);
//	}
//	double difCalculate(double x)
//	{
//		return (1 - (*this)(x))*(1 + (*this)(x));
//	}
//};
//class SoftPlusFunc :public IBaseFunction
//{
//public:
//	double operator()(double x)override
//	{
//		return log(1 + exp(x));
//	}
//	double difCalculate(double x)override
//	{
//		return 1 / (1 + exp(-x));
//	}
//};
//
//class ReLUFunc :public IBaseFunction
//{
//public:
//	double operator()(double x) override
//	{
//		return std::max(0.0, x);
//	}
//	double difCalculate(double x) override
//	{
//
//	}
//};
//
//
//
//class Neiron
//{
//private:
//	vector<double> weights;
//	double localGrad;
//	double difRez;
//	double afRez;
//public:
//	Neiron() :weights(), localGrad(0), difRez(0), afRez(0) {}
//	Neiron(const Neiron&) = default;
//	Neiron(Neiron&&) = default;
//	Neiron(const vector<double>& _weights) :weights(_weights) {}
//	template<typename Func>
//	double OutData(Func& _func, const vector<double>& _inputData) const
//	{
//		return _func(InnerProduct(_inputData, weights));
//	}
//	// ужно додумать
//	void RandomInitWeights()
//	{
//		uniform_int_distribution<int> urd(-10, 10);
//		random_device rd;
//		for (auto& x : weights)
//			x = urd(rd);
//	}
//
//	void InitWeights(const vector<double>& _weights)
//	{
//		weights = _weights;
//	}
//
//	decltype(auto) GetWeights()
//	{
//		return (weights);
//	}
//
//	double GetLocalGrad()const
//	{
//		return localGrad;
//	}
//	void SetLocalGrad(double lg)
//	{
//		localGrad = lg;
//	}
//	void SetDifRez(double df)
//	{
//		difRez = df;
//	}
//	double GetDifRez()const
//	{
//		return difRez;
//	}
//	void SetActiveRez(double afrez)
//	{
//		afRez = afrez;
//	}
//	double GetActiveRez()const
//	{
//		return afRez;
//	}
//};
//class Layer
//{
//private:
//	vector<double>InputData;
//	vector<double>OutputData;
//	vector<Neiron> laer;
//	IBaseFunction* func;
//public:
//	Layer() :InputData(), OutputData(), laer(), func(nullptr) {}
//	Layer(size_t count)
//	{
//		addNeirons(count);
//	}
//	// выставляем входные данные для слоя
//	void SetInputData(const vector<double>& _input)
//	{
//		InputData = _input;
//	}
//	void SetInputData(vector<double>&& _input)
//	{
//		InputData = move(_input);
//	}
//
//	// возвращаем по ссылке выходные данные слоя
//	decltype(auto) GetOutputDataRef()
//	{
//		return (OutputData);
//	}
//
//	// получаем доступ к отдельному нейрону слоя
//	Neiron& operator[](size_t index)
//	{
//		return laer.at(index);
//	}
//
//	// прогоняем данные через весь слой и заносим их в OutputData
//	void Forward()
//	{
//		if (!OutputData.empty())
//		{
//			OutputData.clear();
//		}
//		for (size_t i = 0; i < laer.size(); ++i)
//		{
//			double temp = laer[i].OutData(*func, InputData);
//			laer[i].SetActiveRez(temp);
//			OutputData.emplace_back(temp);
//			laer[i].SetDifRez(laer[i].OutData([&](double a) {return func->difCalculate(a); }, InputData));
//		}
//	}
//
//	/*void addFuncActivated(BaseActivFunction* bsFunc)
//	{
//
//	}*/
//
//	// добавляем новые нейроны в количетсве count
//	void addNeirons(size_t count)
//	{
//		for (size_t i = 0; i < count; ++i)
//		{
//			laer.emplace_back(Neiron());
//		}
//	}
//	// добавляем функцию активации
//	void addActivatedFunction(IBaseFunction* _func)
//	{
//		func = _func;
//	}
//	// добавляем производную функции активации
//	/*void addDifActivatedFunction(IBaseFunction* _diffunc)
//	{
//		difFunc = _diffunc;
//	}*/
//	//количество нейронов
//	size_t GetCountNeiron()const
//	{
//		return laer.size();
//	}
//
//	// инициализация нейронов
//	void InitNeirons(size_t szw)
//	{
//		for (auto& x : laer)
//		{
//			x.GetWeights().resize(szw);
//			x.RandomInitWeights();
//		}
//	}
//
//	// возвращает функцию активации для слоя
//};
//class Net
//{
//private:
//	vector<Layer> layers;
//	double errorNet;
//	vector<double> input;
//	//vector<double> correctOutput;
//	//Layer outLayer;
//	vector<double> output;
//	Model modelNet;
//	size_t szInput;
//	double loss;
//public:
//	Net() = default;
//	void InitInputLayer(size_t _szInput)
//	{
//		szInput = _szInput;
//	}
//	void Forward(const vector<double>& dt)
//	{
//		input = dt;
//		layers[0].SetInputData(dt);
//		layers[0].Forward();
//		for (size_t i = 1; i < layers.size(); ++i)
//		{
//			layers[i].SetInputData(move(layers[i - 1].GetOutputDataRef()));
//			layers[i].Forward();
//		}
//		output = move(layers[layers.size() - 1].GetOutputDataRef());
//	}
//	void Backward()
//	{
//
//	}
//
//	//обучение сети
//	void Training(const vector<double>& trainData, const vector<double>& correctOutput, size_t epoch, double speedLn, double momentum = 0)
//	{
//		uniform_int_distribution<size_t> uid(0, (trainData.size() - 1) / szInput);
//		random_device rd;
//		//correctOutput = corOutput;
//		for (size_t ep = 0; ep < epoch; ++ep)
//		{
//
//				size_t index =uid(rd);
//				size_t pos = index*szInput;
//				// прямой ход
//				Forward(vector<double>(trainData.cbegin() + pos, trainData.cbegin() + szInput + pos));
//
//				// вычисление ошибки сети
//				errorNet = correctOutput[index] - output[0];
//
//				/*if ((ep % 100) == 0&&ep!=0)
//				{
//					cout <<"Error: "<< errorNet << endl;
//				}*/
//
//
//
//				// обратный ход
//
//				//  для выходного слоя устанавливаем локальный градиент каждого нейрона
//
//				for (size_t i = 0; i < layers[layers.size() - 1].GetCountNeiron(); ++i)
//				{
//					layers[layers.size() - 1][i].SetLocalGrad(errorNet*layers[layers.size() - 1][i].GetDifRez());
//				}
//
//				// вычисление локальных градиентов следующих скрытых слоев
//				for (int i = layers.size() - 2; i >= 0; --i)
//				{
//					for (size_t j = 0; j < layers[i].GetCountNeiron(); ++j)
//					{
//						double sum = 0;
//						for (size_t k = 0; k < layers[i + 1].GetCountNeiron(); ++k)
//						{
//							double a = layers[i + 1][k].GetWeights()[j];
//							double b = layers[i + 1][k].GetLocalGrad();
//							sum += a*b;
//						}
//						layers[i][j].SetLocalGrad(sum*layers[i][j].GetDifRez());
//
//					}
//				}
//
//				// корректировка весовых коэффицентов
//
//				// для первого скрытого слоя
//			/*	for (size_t w = 0; w < layers[0][j].GetWeights().size(); ++w)
//					{*/
//				for (size_t j = 0; j < input.size(); ++j)
//				{
//					for (size_t i = 0; i < layers[0].GetCountNeiron(); ++i)
//					{
//						double w = speedLn*input[j] * layers[0][i].GetLocalGrad();
//						layers[0][i].GetWeights()[j] += w;
//					}
//				}
//
//				// для остальных
//
//
//
//				for (size_t l = 1; l < layers.size(); ++l)
//				{
//					for (size_t j = 0; j < layers[l].GetCountNeiron(); ++j)
//					{
//						for (size_t w = 0; w < layers[l][j].GetWeights().size(); ++w)
//						{
//							double dw = speedLn*layers[l - 1][w].GetActiveRez()*layers[l][j].GetLocalGrad();
//							layers[l][j].GetWeights()[w] += dw;
//						}
//					}
//				}
//		
//
//		}
//
//
//	}
//	//void AddLaerAndInit(size_t count)
//	//{
//	//	for (size_t i = 0; i < count; ++i)
//	//	{
//	//		laers.emplace_back(Layer());
//	//	}
//	//}
//
//	// задачется вектор размер которого равен количетсву слоев
//	// каждый элемент вектора это количетсво нейронов в соотвтсвующем слою
//	void SetingNet(const vector<size_t>& stngsHide, const vector<IBaseFunction*>& activFunctions)
//	{
//		// создали все скрытые слои в количетсве stngs.size() с нейронами внутри. количетсво в каждом слою stngs[i]
//		for (size_t i = 0; i < stngsHide.size(); ++i)
//		{
//			layers.emplace_back(Layer(stngsHide[i]));
//			layers[i].addActivatedFunction(activFunctions.at(i));
//		}
//		// связь с входным слоем и первым скрытым слоем
//		if ((szInput == 0))
//			throw exception("Don't init input layer");
//
//		layers[0].InitNeirons(szInput);
//
//		// выставляем связь между остальными скрытыми слоями
//		for (size_t i = 1; i < layers.size(); ++i)
//		{
//			layers[i].InitNeirons(layers[i - 1].GetCountNeiron());
//		}
//
//		// создание выходного слоя
//
//
//	}
//
//	//Надо додумать
//	/*void SaveModel(const string& nameModel = "model")
//	{
//		fstream file;
//		modelNet.SetModelName(nameModel);
//		file.open(nameModel + ".txt");
//	}
//	void InitNet(const string& nameModel)
//	{
//		fstream file;
//		file.open(nameModel);
//		if (!file.is_open())
//		{
//			throw exception("Not found model: may be uncorrect name file");
//		}
//
//	}*/
//	decltype(auto) GetModelNetRef()
//	{
//		return modelNet;
//	}
//
//	// выход сети 
//	decltype(auto) OutNet()const
//	{
//		return (output);
//	}
//
//	// возвращает вектор слоев
//	decltype(auto) GetLayers()
//	{
//		return(layers);
//	}
//};
#define PI 3.1415926535897932384626433832795
using namespace std;
using LaerModel = vector<vector<double>>;
void PTTNormal(vector<double>&vec, double maxEl, double minEl)
{
	double XC = (maxEl + minEl) / 2;
	for (auto& x : vec)
	{
		x = 1 / (1 + exp(-(x - XC)));
	}

}
void NonlinearNormal(vector<double>& vec)
{
	double maxElem = *max_element(vec.begin(), vec.end());
	double minElem = *min_element(vec.begin(), vec.end());
	double XC = (maxElem + minElem) / 2;
	for (auto& x : vec)
	{
		x = 1 / (1 + exp(-(x - XC)));
	}
}
void NonlinearDeNormal(vector<double>& vec, double max, double min)
{
	double XC = (max + min) / 2;
	for (auto& x : vec)
	{
		x = XC - log(1 / x - 1);
	}
}
void LinearNormal(vector<double>& vec, double d1, double d2)
{
	double maxElem = *max_element(vec.begin(), vec.end());
	double minElem = *min_element(vec.begin(), vec.end());
	for (auto& x : vec)
	{
		double t = (x - minElem)*(d2 - d1) / (maxElem - minElem);
		x = t;
	}
}
void LinearDeNormal(vector<double>& vec, double d1, double d2, double minEl, double maxEl)
{
	for (auto& x : vec)
	{
		x = (x - d1)*(maxEl - minEl) / (d2 - d1) + minEl;
	}
}
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
class ReLUFunc :public IBaseFunction
{
public:
	double operator()(double x) override
	{
		return std::max(0.0, x);
	}
	double difCalculate(double x) override
	{
		if (x <= 0)
			return 0;
		else 1;
	}
};
class SinFunc : public IBaseFunction
{
public:
	double operator()(double x)
	{
		return sin(x);
	}
	double difCalculate(double x) override
	{
		return cos(x);
	}
};
class CosFunc :public IBaseFunction
{
public:
	double operator()(double x)
	{
		return cos(x);
	}
	double difCalculate(double x)
	{
		return -sin(x);
	}
};
class GaussFunc :public IBaseFunction
{
public:
	double operator()(double x)
	{
		return exp(-1 * x*x);
	}
	double difCalculate(double x)
	{
		return (*this)(x)*(-2 * x);
	}
};
class TimurFunc :public IBaseFunction
{
public:
	double operator()(double x)
	{
		return sin(x) / x;
	}
	double difCalculate(double x)
	{
		return (cos(x) / x) - (sin(x) / (x*x));
	}
};

class Neiron
{
private:
	vector<double> weights;
	double localGrad;
	double difRez;
	double afRez;
	double delw;
public:
	Neiron() :weights(), localGrad(0), difRez(0), afRez(0), delw(0) {}
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
		//uniform_real_distribution<double> urd(-10, 10);
		normal_distribution<double> dis(0, sqrt(10));
		random_device rd;
		for (auto& x : weights)
			x = dis(rd);
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
	void SetDelW(double w)
	{
		delw = w;
	}
	double GetDelW()const
	{
		return delw;
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
			
			for (size_t ee = 0; ee < correctOutput.size(); ++ee)
			{
				size_t index = uid(rd);
				//	size_t index = ee;
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
				for (int i = layers.size() - 2; i >= 0; --i)
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



				for (size_t i = 0; i < layers[0].GetCountNeiron(); ++i)
				{
					for (size_t j = 0; j < input.size(); ++j)
					{
						double w = speedLn*input[j] * layers[0][i].GetLocalGrad();
						layers[0][i].GetWeights()[j] += w + momentum*layers[0][i].GetDelW();
						layers[0][i].SetDelW(w);
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
							layers[l][j].GetWeights()[w] += dw + momentum*layers[l][j].GetDelW();
							layers[l][j].SetDelW(dw);
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
	l1.addActivatedFunction(new  LogisticFunc());
	l1.addNeirons(3);
	l1[0].InitWeights(vector<double>{1,4});
	l1[1].InitWeights(vector<double>{2,3});
	l1[2].InitWeights(vector<double>{6,4});

	Layer l2;
	l2.addActivatedFunction(new LinearFunc());
	l2.addNeirons(1);
	l2[0].InitWeights(vector<double>{1,3,5});
	//l2[1].InitWeights(vector<double>{1, 1, -4, 8,9,2 });
	//l2[2].InitWeights(vector<double>{3, -2, 1, -9,4,-2});
	
	/*Layer l3;
	l3.addActivatedFunction(new LinearFunc());
	l3.addNeirons(1);
	l3[0].InitWeights(vector<double>{1,2,5});*/


	Net net;
	net.GetLayers().emplace_back(l1);
	net.GetLayers().emplace_back(l2);
	//net.GetLayers().emplace_back(l3);
	net.InitInputLayer(2);
	/*vector<IBaseFunction*> afuns{new LogisticFunc(),new LogisticFunc(),new LogisticFunc(),new SoftPlusFunc() };
	vector<size_t> st{15,1};
	net.SetingNet(st, afuns);*/


	//vector<double> input{ 15,1,15,1,15,1,18,1,28,1,29,1,37,1,37,1,44,1,50,1,50,1,60,1,61,1,64,1,
	//	65,1,65,1,72,1,75,1,75,1,82,1,85,1,91,1,91,1,97,1,98,1,125,1,142,1,142,1,
	//	147,1,147,1,150,1,159,1,165,1,183,1,192,1,195,1,218,1,218,1,219,1,224,1,225,1,
	//	227,1,232,1,232,1,237,1,246,1,258,1,276,1,285,1,300,1,301,1,305,1,312,1,317,1,
	//	338,1,347,1,354,1,357,1,375,1,394,1,513,1,535,1,554,1,591,1,648,1,660,1,
	//	705,1,723,1,756,1,768,1,860,1 };



	//vector<double>d{ 21.66,22.75,22.3,31.25,44.79,40.55,50.25,46.88,52.03,63.47,61.13,81,73.09,79.09,79.51,65.31,71.9,86.1,
	//						94.6,92.5,105,101.7,102.9,110,104.3,134.9,130.68,140.58,155.3,152.2,144.5,142.15,139.81,153.22,145.72,161.1,
	//							174.18,173.03,173.54,178.86,177.68,173.73,159.98,161.29,187.07,176.13,183.4,186.26,189.66,186.09,186.7,186.8,195.1,216.41,
	//						203.23,188.38,189.7,195.31,202.63,224.82,203.3,209.7,233.9,234.7,244.3,231,242.4,230.77,242.57,232.12,246.7};
	vector<double> input{ 2,1 };
	vector<double> d = { 7 };

	////синус
	//double dx=0;
	//for (int i = 0; i < 100; ++i)
	//{
	//	input.emplace_back(dx);
	//	input.emplace_back(1);
	//	d.emplace_back(dx*dx+1);
	//	dx += 0.2;
	//}


	//vector<double> input{ 1,1,2,1,3,1,4,1,5,1,6,1,7,1,8,1,9,1,10,1,11,1,12,1,13,1,14,1,15,1,16,1,17,1,18,1,19,1,20,1,21,1,22,1,23,1,24,1,25,1,26,1,27,1,28,1,29,1 };
	//vector<double> d{ 25.3,23.8,28.7,27.9,27.3,23.8,22.1,23.7,24.1,20.6,27.1,25.4,28.1,23.3,27.4,31,26.9,27.5,25.3,17,21.3,19,23.3,27,23.9,16.5,20.6,24.9,27.3 };
	//vector<double> input{ 1,1,2,1,3,1,4,1,5,1,6,1,7,1 };
	//vector<double> d{ 25.3,23.8,28.7,27.9,27.3,23.8,22.1 };
	
	

	//NonlinearNormal(input);
	//NonlinearNormal(d);

	cout << input.size() << endl;
	cout << d.size() << endl;

	net.Training(input, d, 2000, 0.5);

	int aa = 2;
	for (size_t i = 0; i < d.size(); i++)
	{
		vector<double>t(input.begin() + i *aa, input.begin() + aa + i * aa);
		net.Forward(t);
		cout.precision(30);
		cout.setf(ios_base::fixed);
		cout <<i+1<<") "<<d[i]<<"--"<< net.OutNet()[0] << endl;
	}
	/*vector<size_t> init = { 2,3,1 };
	vector<IBaseFunction*> funcs = { new LogisticFunc(),new LogisticFunc(),new LogisticFunc() };
	Net net;
	net.InitInputLayer(2);
	net.SetingNet(init, funcs);*/

	system("pause");
}