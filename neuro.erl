-module(neuro).
-export([start/0, neuron/5, manager/1]).
-import(data, [get_X/0]).
-define(print(S),io:fwrite(S)). % Макрос вывода русскоязычной строки. Вывод русских строк через format приводит к ошибке.


start()-> 
    Manager = spawn(?MODULE, manager, [[]]),
    Manager ! init,
    X = get_X(),
    complex_work(X, Manager),
%%    Manager ! {work, X1, X2},
    ok111.


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
        O2 = spawn(?MODULE, neuron, initParam()),
        P1 = spawn(?MODULE, neuron, initParam()),


        %  задаем входные веса либо от исходных данных, либо от других нейронов
        I1 ! {addWeight, x1, 1},
        I2 ! {addWeight, x2, 1},

        H1 ! {addWeight, I1, -3.363171846284268},
        H2 ! {addWeight, I1, 3.9442164237878488},
        H3 ! {addWeight, I1, 2.9922150117631654},

        H1 ! {addWeight, I2, 0.7930534650839552},
        H2 ! {addWeight, I2, -0.7758100494565097},
        H3 ! {addWeight, I2, 2.5574413557182742},

        O1 ! {addWeight, H1, 3.5987838698415566},
        O1 ! {addWeight, H2, -4.030738233948683},
        O1 ! {addWeight, H3, 4.598538159596516},

        O2 ! {addWeight, H1, -2.0161703693843216},
        O2 ! {addWeight, H2, 5.10436937137815},
        O2 ! {addWeight, H3, -4.246981385708192},

        P1 ! {addWeight, O1, 1},
        P1 ! {addWeight, O2, 1},

        % нaзначаем смещение
        H1 ! {addBias, -1.2451114742261244}, 
        H2 ! {addBias, -4.670694767699797},
        H3 ! {addBias, -2.244981504083405},

        O1 ! {addBias, -0.7348899787424444},
        O2 ! {addBias, 0.734889978742444},

        %  назначаем нейронам потребителей информации
        I1 ! {addOut, [H1, H2, H3]},
        I2 ! {addOut, [H1, H2, H3]},
        H1 ! {addOut, [O1, O2]},
        H2 ! {addOut, [O1, O2]},
        H3 ! {addOut, [O1, O2]},
        O1 ! {addOut, [P1]},
        O2 ! {addOut, [P1]},

        % у выходного нейрона потребитель менеджер
        P1 ! {addOut, [self()]},

        % определение функции активации. Для простоты проверки просто +1
        % F = fun(X)-> X+1 end,
        Gen_Fun = fun(X) -> X end,
        Hidden_Fun = fun(X) -> math:tanh(X) end,
        Output_Fun = fun(X, Y) -> [math:exp(X)/(math:exp(X) + math:exp(Y)), math:exp(Y)/(math:exp(X) + math:exp(Y))] end ,

        % назначение списку нейронов функции активации
        sndTo([I1, I2], {setFunct, Gen_Fun}),
        sndTo([H1, H2, H3], {setFunct, Hidden_Fun}),
        sndTo([O1, O2], {setFunct, Gen_Fun}),
        sndTo([P1], {setFunct, Output_Fun}),

        % подаем на сеть данные
%%        X1 = 0.743461176196,
%%        X2 = 0.464656328377,
%%        I1!{data, x1, X1},
%%        I2!{data, x2, X2},

        % уходим на ожидание
        manager([I1, I2, H1, H2, H3, O1, O2, P1]);

		{data, _, Data, _} ->
            [C1|[C2|_]] = Data,
            io:format("Prediction is: ~w~n",[index_of_max(C1, C2)]),
%%            self() ! stop,
            manager(Neurons);

       {work, X1, X2} -> 
            [I1|[I2|_]] = Neurons, 
%%            complex_work(get_X(), I1, I2, Neurons),
            
 %%           [X1|[X2|_]] = get_one_X(),
            I1 ! {data, x1, X1, Neurons},
            I2 ! {data, x2, X2, Neurons},
            manager(Neurons);

		stop -> ?print("Менеджер остановлен~n"),
            sndTo(Neurons, stop) end.
   
initParam() -> [maps:new(), [], nill, maps:new(), 0].

neuron(Weights, SndNList, Function, Data, Bias)->
           receive

		   %добавление веса для входящей связи от другого нейрона
		   %веса хранятся как быстрый хеш-словарь, где ключ-ссылка на входящий нейрон, а значение-вес
			{addWeight, FromN, W} ->
				NewWeights = maps:put(FromN, W, Weights),
				neuron(NewWeights, SndNList, Function, Data, Bias);

            %%добавление вмещения
            {addBias, B} ->
                NewBias = B,
				neuron(Weights, SndNList, Function, Data, NewBias);

			% задание данному нейрону списка получателей информации
			{addOut, ToNs}-> neuron(Weights, ToNs, Function, Data, Bias);

			% Назначение нейрону функции активации
			{setFunct, F}-> neuron(Weights, SndNList, F, Data, Bias);

			% прием и обработка данных от нейрона из предыдущего слоя
			{data, FromN, D, Neurons}->
                [P1|_] = lists:reverse(Neurons),
                NewData = maps:put(FromN, D, Data),
                Size = maps:size(Weights),
                case maps:size(NewData) of
                    Size ->
                        if
                            self() == P1 ->
                                Result = other_calculate(Weights, Function, NewData, Bias),
                                sndTo(SndNList, {data, self(), Result, Neurons}),
                                neuron(Weights, SndNList, Function, maps:new(), Bias); 
                            self() /= P1 -> 
                                Result = calculate(Weights, Function, NewData, Bias),
                                sndTo(SndNList, {data, self(), Result, Neurons}),
                                neuron(Weights, SndNList, Function, maps:new(), Bias) 
                        end;
                        % ждем, пока не поступят все данные
                       _ -> neuron(Weights, SndNList, Function, NewData, Bias)
                end;

            % Остановка процесса-нейрона
            stop -> ?print("Нейрон остановлен~n") end.

% вычисление функции активации
calculate(Weights, Function, Data, Bias) ->
    Fun = fun(K, V, AccIn) -> AccIn + V*maps:get(K, Data) end,
    Function(maps:fold(Fun, 0, Weights)) + Bias.

other_calculate(Weights, Function, Data, Bias) ->
    [X1|[X2|_]] = maps:values(Data),
    [Y1|[Y2|_]] = Function(X1, X2),
    io:format("~n~w   ~w~n", [Y1,Y2]),
    Function(X1, X2).
    
% Рассыльщик заданного сообщения по всем процессам в заданном списке
sndTo([],_) -> ok;
sndTo([H|T],M) -> H!M, sndTo(T,M).

index_of_max(X, Y) when X>Y -> 0;
index_of_max(X, Y) -> 1.

complex_work(M, Manager) -> lists:map(fun(X) -> send_one_X(X, Manager) end, M).
send_one_X(X, Manager) ->
    [X1|[X2|_]] = X,
    Manager ! {work, X1, X2}.

    
