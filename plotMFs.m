%% Plot Membership Functions of FIS

function  plotMFs(fis,num_in)

    for i=1:num_in
        subplot(num_in,1,i);
        plotmf(fis,'input',i); grid on;
    end

end