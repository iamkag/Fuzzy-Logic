function plot_response(data, Y, name)
    figure();
    plot(data(:,end-1), data(:,end), 'o', data(:,end-1), Y,'x');
    legend('Data', 'Predictions');
    title(strcat(name,' Original Response vs Predicted Response'));
end