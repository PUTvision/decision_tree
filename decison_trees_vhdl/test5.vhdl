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
			if unsigned(input(1505)) > to_unsigned(19, input'length) then
				splitResult(3) <= '1';
			else
				splitResult(3) <= '0';
			end if;
			if unsigned(input(2095)) > to_unsigned(18, input'length) then
				splitResult(4) <= '1';
			else
				splitResult(4) <= '0';
			end if;
			if unsigned(input(1554)) > to_unsigned(2, input'length) then
				splitResult(5) <= '1';
			else
				splitResult(5) <= '0';
			end if;
			if unsigned(input(1922)) > to_unsigned(4, input'length) then
				splitResult(6) <= '1';
			else
				splitResult(6) <= '0';
			end if;
			if unsigned(input(1040)) > to_unsigned(26, input'length) then
				splitResult(7) <= '1';
			else
				splitResult(7) <= '0';
			end if;
			if unsigned(input(2737)) > to_unsigned(8, input'length) then
				splitResult(8) <= '1';
			else
				splitResult(8) <= '0';
			end if;
			if unsigned(input(1355)) > to_unsigned(21, input'length) then
				splitResult(9) <= '1';
			else
				splitResult(9) <= '0';
			end if;
			if unsigned(input(2005)) > to_unsigned(41, input'length) then
				splitResult(10) <= '1';
			else
				splitResult(10) <= '0';
			end if;
			if unsigned(input(3495)) > to_unsigned(2, input'length) then
				splitResult(11) <= '1';
			else
				splitResult(11) <= '0';
			end if;
			if unsigned(input(2151)) > to_unsigned(16, input'length) then
				splitResult(12) <= '1';
			else
				splitResult(12) <= '0';
			end if;
			if unsigned(input(1388)) > to_unsigned(9, input'length) then
				splitResult(13) <= '1';
			else
				splitResult(13) <= '0';
			end if;
			if unsigned(input(3048)) > to_unsigned(24, input'length) then
				splitResult(14) <= '1';
			else
				splitResult(14) <= '0';
			end if;
			if unsigned(input(2072)) > to_unsigned(8, input'length) then
				splitResult(15) <= '1';
			else
				splitResult(15) <= '0';
			end if;
			if unsigned(input(1945)) > to_unsigned(23, input'length) then
				splitResult(16) <= '1';
			else
				splitResult(16) <= '0';
			end if;
			if unsigned(input(1922)) > to_unsigned(1, input'length) then
				splitResult(17) <= '1';
			else
				splitResult(17) <= '0';
			end if;
			if unsigned(input(1505)) > to_unsigned(8, input'length) then
				splitResult(18) <= '1';
			else
				splitResult(18) <= '0';
			end if;
			if unsigned(input(2798)) > to_unsigned(6, input'length) then
				splitResult(19) <= '1';
			else
				splitResult(19) <= '0';
			end if;
			if unsigned(input(2233)) > to_unsigned(5, input'length) then
				splitResult(20) <= '1';
			else
				splitResult(20) <= '0';
			end if;
			if unsigned(input(2186)) > to_unsigned(11, input'length) then
				splitResult(21) <= '1';
			else
				splitResult(21) <= '0';
			end if;
			if unsigned(input(2240)) > to_unsigned(47, input'length) then
				splitResult(22) <= '1';
			else
				splitResult(22) <= '0';
			end if;
			if unsigned(input(1564)) > to_unsigned(7, input'length) then
				splitResult(23) <= '1';
			else
				splitResult(23) <= '0';
			end if;
			if unsigned(input(2004)) > to_unsigned(16, input'length) then
				splitResult(24) <= '1';
			else
				splitResult(24) <= '0';
			end if;
			if unsigned(input(912)) > to_unsigned(1, input'length) then
				splitResult(25) <= '1';
			else
				splitResult(25) <= '0';
			end if;
			if unsigned(input(564)) > to_unsigned(0, input'length) then
				splitResult(26) <= '1';
			else
				splitResult(26) <= '0';
			end if;
			if unsigned(input(1380)) > to_unsigned(2, input'length) then
				splitResult(27) <= '1';
			else
				splitResult(27) <= '0';
			end if;
			if unsigned(input(1563)) > to_unsigned(5, input'length) then
				splitResult(28) <= '1';
			else
				splitResult(28) <= '0';
			end if;
			if unsigned(input(2190)) > to_unsigned(8, input'length) then
				splitResult(29) <= '1';
			else
				splitResult(29) <= '0';
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
				and
				splitResult(3) = '0'
				and
				splitResult(4) = '0'
				 ) then
					classIndex <= to_unsigned(0, classIndex'length);
				end if;
			if ( 
				splitResult(0) = '0'
				and
				splitResult(1) = '0'
				and
				splitResult(2) = '0'
				and
				splitResult(3) = '0'
				and
				splitResult(4) = '1'
				 ) then
					classIndex <= to_unsigned(1, classIndex'length);
				end if;
			if ( 
				splitResult(0) = '0'
				and
				splitResult(1) = '0'
				and
				splitResult(2) = '0'
				and
				splitResult(3) = '1'
				and
				splitResult(5) = '0'
				 ) then
					classIndex <= to_unsigned(0, classIndex'length);
				end if;
			if ( 
				splitResult(0) = '0'
				and
				splitResult(1) = '0'
				and
				splitResult(2) = '0'
				and
				splitResult(3) = '1'
				and
				splitResult(5) = '1'
				 ) then
					classIndex <= to_unsigned(1, classIndex'length);
				end if;
			if ( 
				splitResult(0) = '0'
				and
				splitResult(1) = '0'
				and
				splitResult(2) = '1'
				and
				splitResult(6) = '0'
				and
				splitResult(7) = '0'
				 ) then
					classIndex <= to_unsigned(0, classIndex'length);
				end if;
			if ( 
				splitResult(0) = '0'
				and
				splitResult(1) = '0'
				and
				splitResult(2) = '1'
				and
				splitResult(6) = '0'
				and
				splitResult(7) = '1'
				 ) then
					classIndex <= to_unsigned(1, classIndex'length);
				end if;
			if ( 
				splitResult(0) = '0'
				and
				splitResult(1) = '0'
				and
				splitResult(2) = '1'
				and
				splitResult(6) = '1'
				and
				splitResult(8) = '0'
				 ) then
					classIndex <= to_unsigned(1, classIndex'length);
				end if;
			if ( 
				splitResult(0) = '0'
				and
				splitResult(1) = '0'
				and
				splitResult(2) = '1'
				and
				splitResult(6) = '1'
				and
				splitResult(8) = '1'
				 ) then
					classIndex <= to_unsigned(0, classIndex'length);
				end if;
			if ( 
				splitResult(0) = '0'
				and
				splitResult(1) = '1'
				and
				splitResult(9) = '0'
				and
				splitResult(10) = '0'
				and
				splitResult(11) = '0'
				 ) then
					classIndex <= to_unsigned(1, classIndex'length);
				end if;
			if ( 
				splitResult(0) = '0'
				and
				splitResult(1) = '1'
				and
				splitResult(9) = '0'
				and
				splitResult(10) = '0'
				and
				splitResult(11) = '1'
				 ) then
					classIndex <= to_unsigned(0, classIndex'length);
				end if;
			if ( 
				splitResult(0) = '0'
				and
				splitResult(1) = '1'
				and
				splitResult(9) = '0'
				and
				splitResult(10) = '1'
				and
				splitResult(12) = '0'
				 ) then
					classIndex <= to_unsigned(0, classIndex'length);
				end if;
			if ( 
				splitResult(0) = '0'
				and
				splitResult(1) = '1'
				and
				splitResult(9) = '0'
				and
				splitResult(10) = '1'
				and
				splitResult(12) = '1'
				 ) then
					classIndex <= to_unsigned(1, classIndex'length);
				end if;
			if ( 
				splitResult(0) = '0'
				and
				splitResult(1) = '1'
				and
				splitResult(9) = '1'
				and
				splitResult(13) = '0'
				and
				splitResult(14) = '0'
				 ) then
					classIndex <= to_unsigned(0, classIndex'length);
				end if;
			if ( 
				splitResult(0) = '0'
				and
				splitResult(1) = '1'
				and
				splitResult(9) = '1'
				and
				splitResult(13) = '0'
				and
				splitResult(14) = '1'
				 ) then
					classIndex <= to_unsigned(1, classIndex'length);
				end if;
			if ( 
				splitResult(0) = '0'
				and
				splitResult(1) = '1'
				and
				splitResult(9) = '1'
				and
				splitResult(13) = '1'
				and
				splitResult(15) = '0'
				 ) then
					classIndex <= to_unsigned(1, classIndex'length);
				end if;
			if ( 
				splitResult(0) = '0'
				and
				splitResult(1) = '1'
				and
				splitResult(9) = '1'
				and
				splitResult(13) = '1'
				and
				splitResult(15) = '1'
				 ) then
					classIndex <= to_unsigned(0, classIndex'length);
				end if;
			if ( 
				splitResult(0) = '1'
				and
				splitResult(16) = '0'
				and
				splitResult(17) = '0'
				and
				splitResult(18) = '0'
				and
				splitResult(19) = '0'
				 ) then
					classIndex <= to_unsigned(0, classIndex'length);
				end if;
			if ( 
				splitResult(0) = '1'
				and
				splitResult(16) = '0'
				and
				splitResult(17) = '0'
				and
				splitResult(18) = '0'
				and
				splitResult(19) = '1'
				 ) then
					classIndex <= to_unsigned(1, classIndex'length);
				end if;
			if ( 
				splitResult(0) = '1'
				and
				splitResult(16) = '0'
				and
				splitResult(17) = '0'
				and
				splitResult(18) = '1'
				and
				splitResult(20) = '0'
				 ) then
					classIndex <= to_unsigned(1, classIndex'length);
				end if;
			if ( 
				splitResult(0) = '1'
				and
				splitResult(16) = '0'
				and
				splitResult(17) = '0'
				and
				splitResult(18) = '1'
				and
				splitResult(20) = '1'
				 ) then
					classIndex <= to_unsigned(0, classIndex'length);
				end if;
			if ( 
				splitResult(0) = '1'
				and
				splitResult(16) = '0'
				and
				splitResult(17) = '1'
				and
				splitResult(21) = '0'
				and
				splitResult(22) = '0'
				 ) then
					classIndex <= to_unsigned(1, classIndex'length);
				end if;
			if ( 
				splitResult(0) = '1'
				and
				splitResult(16) = '0'
				and
				splitResult(17) = '1'
				and
				splitResult(21) = '0'
				and
				splitResult(22) = '1'
				 ) then
					classIndex <= to_unsigned(0, classIndex'length);
				end if;
			if ( 
				splitResult(0) = '1'
				and
				splitResult(16) = '0'
				and
				splitResult(17) = '1'
				and
				splitResult(21) = '1'
				 ) then
					classIndex <= to_unsigned(0, classIndex'length);
				end if;
			if ( 
				splitResult(0) = '1'
				and
				splitResult(16) = '1'
				and
				splitResult(23) = '0'
				and
				splitResult(24) = '0'
				and
				splitResult(25) = '0'
				 ) then
					classIndex <= to_unsigned(0, classIndex'length);
				end if;
			if ( 
				splitResult(0) = '1'
				and
				splitResult(16) = '1'
				and
				splitResult(23) = '0'
				and
				splitResult(24) = '0'
				and
				splitResult(25) = '1'
				 ) then
					classIndex <= to_unsigned(1, classIndex'length);
				end if;
			if ( 
				splitResult(0) = '1'
				and
				splitResult(16) = '1'
				and
				splitResult(23) = '0'
				and
				splitResult(24) = '1'
				and
				splitResult(26) = '0'
				 ) then
					classIndex <= to_unsigned(1, classIndex'length);
				end if;
			if ( 
				splitResult(0) = '1'
				and
				splitResult(16) = '1'
				and
				splitResult(23) = '0'
				and
				splitResult(24) = '1'
				and
				splitResult(26) = '1'
				 ) then
					classIndex <= to_unsigned(0, classIndex'length);
				end if;
			if ( 
				splitResult(0) = '1'
				and
				splitResult(16) = '1'
				and
				splitResult(23) = '1'
				and
				splitResult(27) = '0'
				and
				splitResult(28) = '0'
				 ) then
					classIndex <= to_unsigned(0, classIndex'length);
				end if;
			if ( 
				splitResult(0) = '1'
				and
				splitResult(16) = '1'
				and
				splitResult(23) = '1'
				and
				splitResult(27) = '0'
				and
				splitResult(28) = '1'
				 ) then
					classIndex <= to_unsigned(1, classIndex'length);
				end if;
			if ( 
				splitResult(0) = '1'
				and
				splitResult(16) = '1'
				and
				splitResult(23) = '1'
				and
				splitResult(27) = '1'
				and
				splitResult(29) = '0'
				 ) then
					classIndex <= to_unsigned(1, classIndex'length);
				end if;
			if ( 
				splitResult(0) = '1'
				and
				splitResult(16) = '1'
				and
				splitResult(23) = '1'
				and
				splitResult(27) = '1'
				and
				splitResult(29) = '1'
				 ) then
					classIndex <= to_unsigned(0, classIndex'length);
				end if;
		end if;
	end if;
end process decideClass;
