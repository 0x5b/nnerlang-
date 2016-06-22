-module(main).

-import(data, [get_X/0, get_Y/0]).
-export([start/0, run/0]).

run() ->
    {Time,_} = timer:tc(main, start, []),
    io:format("  Elapsed time: ~w sec~n", [Time/1000000]).

start() ->
    X = get_X(),
    Y = get_Y(),

    Model = multilayer_perceptron(),
    print_format(model, Model),

    {Model1, Count, Loss} = train_network(Model, X, Y, 0.01, 0, 0.0000001, 0),
    print_format(model1, Model1),

    io:format("  Iteration count: ~w~n", [Count]),
    io:format("  Minimal loss: ~w~n", [Loss]).

multilayer_perceptron() ->
    W1 = [[random:uniform(), random:uniform(), random:uniform()],
          [random:uniform(), random:uniform(), random:uniform()]],

    B1 = [0.0, 0.0, 0.0],

    W2 = [[random:uniform(), random:uniform()],
          [random:uniform(), random:uniform()],
          [random:uniform(), random:uniform()]],

    B2 = [0.0, 0.0],

    {W1, B1, W2, B2}.


forward_propagation(X, Model) ->
    {W1, B1, W2, B2} = Model,

    Z1 = sumWithBias(mul(X, W1), B1),
    A1 = tanhMatrix(Z1),

    Z2 = sumWithBias(mul(A1, W2), B2),
    Exp_scores = expMatrix(Z2),
    Y_hat = output_func(Exp_scores),
    [Y_hat, Z1, A1, Z2].
    

backpropagation(X, Y, Y_hat, A1, Model) ->
    {_, _, W2, _} = Model,

    Delta3 = deltaMatrix(Y_hat, Y),
    DW2 = mul(trans(A1), Delta3),
    DB2 = sumByStolb(Delta3),

    Delta2 = scalarMulMatrix(mul(Delta3, trans(W2)), subFrom1Matrix(power2Matrix(A1))),
    DW1 = mul(trans(X), Delta2),
    DB1 = sumByStolb(Delta2),
    {DW1, DB1, DW2, DB2}.


calculate_loss(X, Y, Model, Y_hat) ->
    {W1, _, W2, _} = Model,

    Correct_logprobs = logCorrAns(Y_hat, Y),
    Data_loss = lists:sum(Correct_logprobs),
    Reg_loss = Data_loss + 0.01 / 2 * (sumElemMatrix(power2Matrix(W1)) + sumElemMatrix(power2Matrix(W2))),
    1/length(X)*Reg_loss.


%%Тренировка проходит 1 раз (просто для проверки, после проверки сделаю в бесконечном цикле до определенного уровня)
train_network(Model, X, Y, Learn_rate, Last_loss, Min_diff, Count) ->
    {W1, B1, W2, B2} = Model,

    [Y_hat, _, A1, _] = forward_propagation(X, Model),

    {DW1, DB1, DW2, DB2} = backpropagation(X, Y, Y_hat, A1, Model),

    RegDW1 = sumMatrix(DW1, mulMatrixN(DW1, 0.01)),
    RegDW2 = sumMatrix(DW2, mulMatrixN(DW2, 0.01)),

    NewW1 = sumMatrix(W1, mulMatrixN(RegDW1, -Learn_rate)),
    NewB1 = sumMLists(B1, mulN(DB1, -Learn_rate)),
    NewW2 = sumMatrix(W2, mulMatrixN(RegDW2, -Learn_rate)),
    NewB2 = sumMLists(B2, mulN(DB2, -Learn_rate)),

    Current_loss = calculate_loss(X, Y, Model, Y_hat),
    NewCount = Count + 1,
    if abs(Current_loss - Last_loss) > Min_diff ->
            New_loss = Current_loss,
            if
                Count rem 50 == 0 ->
                    io:format("  Loss after iteration ~w: ~w~n", [Count, Current_loss]);
                Count rem 50 /= 0 ->
                    _ = Count
            end,
            NewModel = {NewW1, NewB1, NewW2, NewB2},
            train_network(NewModel, X, Y, Learn_rate, New_loss, Min_diff, NewCount);
    abs(Current_loss - Last_loss) =< Min_diff->
           {{NewW1, NewB1, NewW2, NewB2}, NewCount, Current_loss}
        end.

    
%% Matrix список списков одинаковой длины
trans(Matrix)-> N=length(lists:nth(1,Matrix)), %количество столбцов
				doN(1,N,Matrix,[]).

doN(K,N,_,TMatrix) when K>N ->
    lists:reverse(TMatrix);
doN(K,N,Matrix,TMatrix) ->
    doN(K+1,N,Matrix,[getK(K,Matrix,[])|TMatrix]).

getK(_,[],Akk) ->
    lists:reverse(Akk);
getK(K,[H|T],Akk) ->
    getK(K,T,[lists:nth(K,H)|Akk]).


%%----------------------------------------------------------------------------
mul(M1,M2) -> 
    N=length(lists:nth(1,M2)), %количество столбцов во второй матрице
	doMul(M1,M2,N,[]).

doMul([],_,_,Akk) ->
    lists:reverse(Akk);
doMul([H|T],M2,N,Akk) ->
    doMul(T,M2,N, [doMul2(H,M2,1,N,[]) | Akk]).

doMul2(_,_,K,N,Akk) when K>N ->
    lists:reverse(Akk);
doMul2(H,M2,K,N,Akk) ->
    doMul2(H, M2, K+1, N, [lists:sum(lists:zipwith(fun(X,Y)->X*Y end,H,getKth(K,M2,[]))) | Akk]).

% взять К-ый столбец
getKth(_,[],Akk) ->
    lists:reverse(Akk);
getKth(K,[H|T],Akk) ->
    getKth(K,T,[lists:nth(K,H)|Akk]).


%%Сложение взвешенной суммы каждого для каждого нейрона с порогом bi
sumWithBias(M, B) -> lists:map(fun(X)->sumLists(X, B) end, M).
sumLists(L1, L2) -> lists:zipwith(fun(X, Y)->X+Y end, L1, L2).


%%Гиперболический тангенс для каждого элемента матрицы (для каждого нейрона скрытого слоя)
tanhMatrix(M) -> lists:map(fun(X)->tanhList(X) end, M).
tanhList(L) -> lists:map(fun(X)-> math:tanh(X) end, L).


%%e^x для каждого элемента в матрице
expMatrix(M) -> lists:map(fun(X)->expList(X) end, M).
expList(L) -> lists:map(fun(X)->math:exp(X) end, L).


%%функция активации выходного слоя e^x1/(e^x1+e^x2)
output_func(M) -> lists:map(fun(X)->softmax(X) end, M).
softmax(Exp_scores) ->
    [P1|[P2|_]] = Exp_scores,
    [P1/(P1+P2), P2/(P1+P2)].


%%функция высчитывает y'- y (y' - решение сети, y - правильные ответы)
deltaMatrix(M1, M2) -> lists:zipwith(fun(X,Y)->delta(X,Y) end, M1,M2).
delta(List, Y) when Y==0 ->
    [X1, X2] = List,
    [X1-1,X2];
delta(List, Y) ->
    [X1, X2] = List,
    [X1,X2-1].


logCorrAns(Matrix, A) -> lists:zipwith(fun(X,Y)->logLists(X, Y) end, Matrix, A).
logLists(List, Y) when Y==0 ->
    [X1, _] = List,
    -math:log(X1);
logLists(List, Y) ->
    [_, X2] = List,
    -math:log(X2).


%%функция возводит каждый элемент матрицы в квадрат
power2Matrix(M) -> lists:map(fun(X)->power2List(X) end, M).
power2List(L) -> lists:map(fun(X)->X*X end, L).


%%функция вычитает матрицу из единичной матрицы
subFrom1Matrix(M) -> lists:map(fun(X)->subFrom1List(X) end, M).
subFrom1List(L) -> lists:map(fun(X)->1-X end, L).


%%функция cкалярно перемножает строки 2х матриц
scalarMulMatrix(M1, M2) -> lists:zipwith(fun(X,Y)->scalarMul(X,Y) end, M1, M2).
scalarMul(L1, L2) -> lists:zipwith(fun(X,Y)->X*Y end, L1, L2).


%%умножение матрицы на число
mulMatrixN(M, N) -> lists:map(fun(X)->mulN(X, N) end, M).
mulN(L, N) -> lists:map(fun(X)->X*N end, L).


%%сложение матриц
sumMatrix(M1, M2) -> lists:zipwith(fun(X,Y)->sumMLists(X,Y) end, M1, M2).
sumMLists(L1, L2) -> lists:zipwith(fun(X,Y)->X+Y end, L1, L2).

%%сумма элементов столбцов
sumByStolb(Matrix)-> 
    N=length(lists:nth(1,Matrix)), %количество столбцов
    doSumStolb(1,N,Matrix,[]).

doSumStolb(K,N,_,Akk) when K>N ->
    lists:reverse(Akk);
doSumStolb(K,N,Matrix,Akk) ->
    doSumStolb(K+1,N,Matrix,[lists:sum(getK(K,Matrix,[]))|Akk]).


sumElemMatrix(Matrix) -> lists:sum(lists:map(fun(X)->lists:sum(X) end, Matrix)).
%%-----------------------------------------------------------------------------
print_format(What ,{W1, B1, W2, B2}) ->
    io:format("  ~n  ~p:~n  w1 = ~w~n  b1 = ~w~n  w2 = ~w~n  b2 = ~w~n", [What, W1, B1, W2, B2]).
