compare : process(clk)
begin
	if clk='1' and clk'event then
		if rst='1' then
		elsif en='1' then
			if unsigned(input(2092)) > to_unsigned(6, input'length) then
				splitResult(0) <= '1';
			else
				splitResult(0) <= '0';
			end if;
			if unsigned(input(2091)) > to_unsigned(6, input'length) then
				splitResult(1) <= '1';
			else
				splitResult(1) <= '0';
			end if;
			if unsigned(input(1565)) > to_unsigned(11, input'length) then
				splitResult(2) <= '1';
			else
				splitResult(2) <= '0';
			end if;
			if unsigned(input(1355)) > to_unsigned(21, input'length) then
				splitResult(3) <= '1';
			else
				splitResult(3) <= '0';
			end if;
			if unsigned(input(1945)) > to_unsigned(23, input'length) then
				splitResult(4) <= '1';
			else
				splitResult(4) <= '0';
			end if;
			if unsigned(input(1922)) > to_unsigned(1, input'length) then
				splitResult(5) <= '1';
			else
				splitResult(5) <= '0';
			end if;
			if unsigned(input(1564)) > to_unsigned(7, input'length) then
				splitResult(6) <= '1';
			else
				splitResult(6) <= '0';
			end if;
		end if;
	end if;
end process compare;

decideClass : process(clk)
begin
	if clk='1' and clk'event then
		if rst='1' then
		
		elsif en='1' then
			if ( 
				splitResult(0) = '0'
				and
				splitResult(1) = '0'
				and
				splitResult(2) = '0'
				 ) then
					classIndex <= to_unsigned(0, classIndex'length);
				end if;
			if ( 
				splitResult(0) = '0'
				and
				splitResult(1) = '0'
				and
				splitResult(2) = '1'
				 ) then
					classIndex <= to_unsigned(0, classIndex'length);
				end if;
			if ( 
				splitResult(0) = '0'
				and
				splitResult(1) = '1'
				and
				splitResult(3) = '0'
				 ) then
					classIndex <= to_unsigned(1, classIndex'length);
				end if;
			if ( 
				splitResult(0) = '0'
				and
				splitResult(1) = '1'
				and
				splitResult(3) = '1'
				 ) then
					classIndex <= to_unsigned(0, classIndex'length);
				end if;
			if ( 
				splitResult(0) = '1'
				and
				splitResult(4) = '0'
				and
				splitResult(5) = '0'
				 ) then
					classIndex <= to_unsigned(1, classIndex'length);
				end if;
			if ( 
				splitResult(0) = '1'
				and
				splitResult(4) = '0'
				and
				splitResult(5) = '1'
				 ) then
					classIndex <= to_unsigned(1, classIndex'length);
				end if;
			if ( 
				splitResult(0) = '1'
				and
				splitResult(4) = '1'
				and
				splitResult(6) = '0'
				 ) then
					classIndex <= to_unsigned(0, classIndex'length);
				end if;
			if ( 
				splitResult(0) = '1'
				and
				splitResult(4) = '1'
				and
				splitResult(6) = '1'
				 ) then
					classIndex <= to_unsigned(1, classIndex'length);
				end if;
		end if;
	end if;
end process decideClass;
