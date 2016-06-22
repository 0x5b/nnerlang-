-module(neuro).
-export([start/0, neuron/4, manager/1]).
%%-import(data, [get_one_X/0]).
-define(print(S),io:fwrite(S)). % Макрос вывода русскоязычной строки. Вывод русских строк через format приводит к ошибке.


start()-> 
    Manager = spawn(?MODULE, manager, [[]]),
    Manager!init,
	ok.


manager(Neurons) ->
   receive
      init->
        % запускаем нейроны
        I1 = spawn(?MODULE, neuron, initParam()),
        I2 = spawn(?MODULE, neuron, initParam()),
%%        register(i1, spawn(?MODULE, neuron, initParam())),
%%        register(i2, spawn(?MODULE, neuron, initParam())),
        H1 = spawn(?MODULE, neuron, initParam()),
        H2 = spawn(?MODULE, neuron, initParam()),
        H3 = spawn(?MODULE, neuron, initParam()),
        O1 = spawn(?MODULE, neuron, initParam()),

        %  задаем входные веса либо от исходных данных, либо от других нейронов
        I1 ! {addWeight, x1, 1},
        I2 ! {addWeight, x2, 1},
        H1 ! {addWeight, I1, 1},
        H1 ! {addWeight, I2, 1},
        H2 ! {addWeight, I1, 1},
        H2 ! {addWeight, I2, 1},
        H3 ! {addWeight, I1, 1},
        H3 ! {addWeight, I2, 1},
        O1 ! {addWeight, H1, 1},
        O1 ! {addWeight, H2, 1},
        O1 ! {addWeight, H3, 1},

        %  назначаем нейронам потребителей информации
        I1 ! {addOut, [H1, H2, H3]},
        I2 ! {addOut, [H1, H2, H3]},
        H1 ! {addOut, [O1]},
        H2 ! {addOut, [O1]},
        H3 ! {addOut, [O1]},

        % у выходного нейрона потребитель менеджер
        O1!{addOut, [self()]},

        % определение функции активации. Для простоты проверки просто +1
        % F = fun(X)-> X+1 end,
        Input_Fun = fun(X) -> X-1 end,
        Hidden_Fun = fun(X) -> math:tanh(X) end,
        Output_Fun = fun(X) -> 1/(1 + math:exp(-X)) end,

        % назначение списку нейронов функции активации
        sndTo([I1, I2], {setFunct, Input_Fun}),
        sndTo([H1, H2, H3], {setFunct, Hidden_Fun}),
        sndTo([O1], {setFunct, Output_Fun}),

        % подаем на сеть данные
        X1 = 0.743461176196,
        X2 = 0.464656328377,
        I1!{data, x1, X1},
        I2!{data, x2, X2},

        % уходим на ожидание
        manager([I1, I2, H1, H2, H3, O1]);

		{data, _, Data} ->
            io:format("Result is: ~w~n",[Data]),
            self()!stop,
            manager(Neurons);

        {train_data, I1, I2, X1, X2} -> 
            I1!{data, x1, X1},
            I2!{data, x2, X2};

		stop -> ?print("Менеджер остановлен~n"),
            sndTo(Neurons, stop) end.
   
initParam() -> [maps:new(), [], nill, maps:new()].

neuron(Weights, SndNList, Function, Data)->
           receive

		   %добавление веса для входящей связи от другого нейрона
		   %веса хранятся как быстрый хеш-словарь, где ключ-ссылка на входящий нейрон, а значение-вес
			{addWeight, FromN, W} ->
				NewWeights = maps:put(FromN, W, Weights),
				neuron(NewWeights, SndNList, Function, Data);

			% задание данному нейрону списка получателей информации
			{addOut, ToNs}-> neuron(Weights, ToNs, Function, Data);

			% Назначение нейрону функции активации
			{setFunct, F}-> neuron(Weights, SndNList, F, Data);

			% прием и обработка данных от нейрона из предыдущего слоя
			{data, FromN, D}->
                NewData = maps:put(FromN, D, Data),
                Size = maps:size(Weights),
                case maps:size(NewData) of
                    Size ->
                        Result = calculate(Weights, Function, NewData),
                        sndTo(SndNList, {data, self(), Result}),
                        % когда результаты вычислены и отправлены, забываем текущие входные данные
                        neuron(Weights, SndNList, Function, maps:new()); 
                        % ждем, пока не поступят все данные
                        _ -> neuron(Weights, SndNList, Function, NewData)
                end;

            % Остановка процесса-нейрона
            stop -> ?print("Нейрон остановлен~n") end.

% вычисление функции активации
calculate(Weights, Function, Data) ->
    Fun = fun(K, V, AccIn) -> AccIn + V * maps:get(K, Data) end,
    Function(maps:fold(Fun, 0, Weights)).
								   
% Рассыльщик заданного сообщения по всем процессам в заданном списке
sndTo([],_) -> ok;
sndTo([H|T],M) -> H!M, sndTo(T,M).

